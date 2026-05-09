"""
Training Loop for Mamba Anomaly Classifier

Self-supervised training using synthetic anomaly injection.
The model learns to classify windows as normal (0) or anomalous (1).

Training Strategy:
    1. Load raw univariate signal (no feature engineering)
    2. Create sliding windows
    3. For each batch:
       - Keep half as clean (label=0)
       - Inject synthetic anomalies into the other half (label=1)
    4. Train classifier with Binary Cross Entropy loss
    5. Validate on held-out windows (with fresh anomaly injection)

Key Features:
    - Early stopping with patience
    - Model checkpointing (best validation loss)
    - Learning rate scheduling
    - Gradient clipping
    - Mixed precision training (optional)
    - Comprehensive logging
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

# Local imports
from models.mamba_encoder import MambaAnomalyClassifier, create_mamba_classifier, count_parameters
from models.anomaly_injector import AnomalyInjector, AnomalyConfig
from utils.data_loader import DataLoader, ChannelData


# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TrainingConfig:
    """Training hyperparameters and settings."""
    
    def __init__(
        self,
        # Data
        window_size: int = 100,
        step: int = 1,
        val_split: float = 0.1,
        
        # Model
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
        pool_strategy: str = 'last',
        
        # Training
        epochs: int = 50,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        grad_clip: float = 1.0,
        
        # Anomaly injection
        injection_ratio: float = 0.5,
        
        # Early stopping
        patience: int = 10,
        min_delta: float = 1e-4,
        
        # Checkpointing
        save_dir: str = 'results',
        save_best: bool = True,
        
        # Misc
        seed: Optional[int] = 42,
        verbose: bool = True,
    ):
        self.window_size = window_size
        self.step = step
        self.val_split = val_split
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_layers = n_layers
        self.dropout = dropout
        self.pool_strategy = pool_strategy
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        
        self.injection_ratio = injection_ratio
        
        self.patience = patience
        self.min_delta = min_delta
        
        self.save_dir = save_dir
        self.save_best = save_best
        
        self.seed = seed
        self.verbose = verbose
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return vars(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        return cls(**d)


# ============================================================================
# Training Utilities
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class TrainingHistory:
    """Track training metrics over epochs."""
    
    def __init__(self):
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.train_acc: List[float] = []
        self.val_acc: List[float] = []
        self.learning_rates: List[float] = []
        self.epoch_times: List[float] = []
    
    def update(
        self,
        train_loss: float,
        val_loss: float,
        train_acc: float,
        val_acc: float,
        lr: float,
        epoch_time: float,
    ):
        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(train_acc)
        self.val_acc.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)
    
    def to_dict(self) -> Dict:
        return {
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'train_acc': self.train_acc,
            'val_acc': self.val_acc,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times,
        }


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Core Training Functions
# ============================================================================

def prepare_data(
    channel_data: ChannelData,
    config: TrainingConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training and validation windows from channel data.
    
    Returns:
        train_windows: Shape (n_train, window_size)
        val_windows: Shape (n_val, window_size)
    """
    # Create sliding windows from normalized training data
    windows = DataLoader.create_windows(
        channel_data.train_normalized,
        window_size=config.window_size,
        step=config.step,
    )
    
    # Shuffle windows
    indices = np.random.permutation(len(windows))
    windows = windows[indices]
    
    # Split into train and validation
    n_val = int(len(windows) * config.val_split)
    if n_val < 1:
        n_val = 1
    
    val_windows = windows[:n_val]
    train_windows = windows[n_val:]
    
    print(f"Data prepared:")
    print(f"  Training windows: {len(train_windows)}")
    print(f"  Validation windows: {len(val_windows)}")
    
    return train_windows, val_windows


def train_epoch(
    model: MambaAnomalyClassifier,
    train_windows: np.ndarray,
    injector: AnomalyInjector,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    config: TrainingConfig,
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        avg_loss: Average loss over batches
        accuracy: Classification accuracy
    """
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0
    
    # Shuffle training data
    indices = np.random.permutation(len(train_windows))
    
    for i in range(0, len(indices), config.batch_size):
        batch_indices = indices[i:i + config.batch_size]
        if len(batch_indices) < 2:
            continue  # Skip tiny batches
        
        # Get batch windows
        batch_windows = train_windows[batch_indices]
        
        # Inject anomalies (self-supervised labels)
        modified_windows, labels = injector.inject_batch(
            batch_windows,
            injection_ratio=config.injection_ratio,
        )
        
        # Convert to tensors
        x = torch.tensor(modified_windows, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        if config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += len(y)
        n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_samples, 1)
    
    return avg_loss, accuracy


@torch.no_grad()
def validate_epoch(
    model: MambaAnomalyClassifier,
    val_windows: np.ndarray,
    injector: AnomalyInjector,
    criterion: nn.Module,
    config: TrainingConfig,
) -> Tuple[float, float]:
    """
    Validate for one epoch.
    
    Returns:
        avg_loss: Average loss over batches
        accuracy: Classification accuracy
    """
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0
    
    for i in range(0, len(val_windows), config.batch_size):
        batch_windows = val_windows[i:i + config.batch_size]
        if len(batch_windows) < 1:
            continue
        
        # Inject anomalies (same process as training for fair comparison)
        modified_windows, labels = injector.inject_batch(
            batch_windows,
            injection_ratio=config.injection_ratio,
        )
        
        # Convert to tensors
        x = torch.tensor(modified_windows, dtype=torch.float32).to(DEVICE)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(DEVICE)
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits, y)
        
        # Track metrics
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (preds == y).sum().item()
        total_samples += len(y)
        n_batches += 1
    
    avg_loss = total_loss / max(n_batches, 1)
    accuracy = total_correct / max(total_samples, 1)
    
    # Synchronize CUDA to catch async errors early
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()
    
    return avg_loss, accuracy


def train(
    channel_id: str,
    data_dir: str = '../data',
    config: Optional[TrainingConfig] = None,
) -> Tuple[MambaAnomalyClassifier, TrainingHistory, TrainingConfig]:
    """
    Full training pipeline for a single channel.
    
    Args:
        channel_id: Channel to train on (e.g., 'A-1')
        data_dir: Path to data directory
        config: Training configuration (uses defaults if None)
        
    Returns:
        model: Trained model
        history: Training history
        config: Configuration used
    """
    if config is None:
        config = TrainingConfig()
    
    # Set seed for reproducibility
    if config.seed is not None:
        set_seed(config.seed)
    
    print(f"\n{'='*60}")
    print(f"Training Mamba Anomaly Classifier")
    print(f"Channel: {channel_id}")
    print(f"Device: {DEVICE}")
    print(f"{'='*60}\n")
    
    # Load data
    loader = DataLoader(data_dir=data_dir)
    channel_data = loader.load_channel(channel_id)
    
    # Prepare train/val split
    train_windows, val_windows = prepare_data(channel_data, config)
    
    # Create model
    model = MambaAnomalyClassifier(
        input_dim=1,
        d_model=config.d_model,
        d_state=config.d_state,
        d_conv=config.d_conv,
        expand=config.expand,
        n_layers=config.n_layers,
        dropout=config.dropout,
        pool_strategy=config.pool_strategy,
    ).to(DEVICE)
    
    print(f"\nModel created:")
    print(f"  Parameters: {count_parameters(model):,}")
    print(f"  Architecture: {config.n_layers} Mamba blocks, d_model={config.d_model}")
    
    # Create anomaly injector
    injector = AnomalyInjector(seed=config.seed)
    
    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.patience,
        min_delta=config.min_delta,
    )
    
    # Training history
    history = TrainingHistory()
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    
    # Create save directory
    save_dir = Path(config.save_dir) / channel_id
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nStarting training for {config.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        
        try:
            # Train
            train_loss, train_acc = train_epoch(
                model, train_windows, injector, optimizer, criterion, config
            )
            
            # Validate
            val_loss, val_acc = validate_epoch(
                model, val_windows, injector, criterion, config
            )
            
            # Clear CUDA cache after each epoch
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            if 'CUDA' in str(e) or 'out of memory' in str(e):
                print(f"\n⚠️  CUDA error at epoch {epoch + 1}: {type(e).__name__}")
                print(f"   {str(e)[:100]}...")
                
                # Clear CUDA state
                if DEVICE.type == 'cuda':
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # If we have a saved checkpoint, stop gracefully
                if best_val_loss < float('inf'):
                    print(f"   Best model saved at epoch {best_epoch} (val_loss={best_val_loss:.4f})")
                    print("   Stopping training but checkpoint is available.")
                    print("\n" + "="*60)
                    break
                else:
                    print("   No checkpoint available. Re-raising error.")
                    raise
            else:
                raise
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Track history
        epoch_time = time.time() - epoch_start
        history.update(train_loss, val_loss, train_acc, val_acc, current_lr, epoch_time)
        
        # Print progress
        if config.verbose:
            print(
                f"Epoch {epoch+1:3d}/{config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Train Acc: {train_acc:.3f} | "
                f"Val Acc: {val_acc:.3f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {epoch_time:.1f}s"
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            if config.save_best:
                checkpoint_path = save_dir / 'best_model.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': best_model_state,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'config': config.to_dict(),
                }, checkpoint_path)
                if config.verbose:
                    print(f"  → Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Early stopping check
        if early_stopping(val_loss):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"\nLoaded best model with val_loss: {best_val_loss:.4f}")
    
    # Save final results
    results = {
        'channel_id': channel_id,
        'best_val_loss': best_val_loss,
        'final_train_loss': history.train_loss[-1],
        'final_val_loss': history.val_loss[-1],
        'final_train_acc': history.train_acc[-1],
        'final_val_acc': history.val_acc[-1],
        'total_epochs': len(history.train_loss),
        'config': config.to_dict(),
        'history': history.to_dict(),
    }
    
    results_path = save_dir / 'training_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"  Best validation loss: {best_val_loss:.4f}")
    print(f"  Results saved to: {save_dir}")
    
    return model, history, config


# ============================================================================
# CLI Interface
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Mamba Anomaly Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required
    parser.add_argument(
        '--channel', '-c',
        type=str,
        required=True,
        help='Channel ID to train on (e.g., A-1)',
    )
    
    # Data
    parser.add_argument('--data-dir', type=str, default='../data', help='Data directory')
    parser.add_argument('--window-size', type=int, default=100, help='Window size')
    parser.add_argument('--step', type=int, default=1, help='Window step size')
    parser.add_argument('--val-split', type=float, default=0.1, help='Validation split ratio')
    
    # Model
    parser.add_argument('--d-model', type=int, default=64, help='Model dimension')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of Mamba blocks')
    parser.add_argument('--d-state', type=int, default=16, help='SSM state dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--pool', type=str, default='last', 
                       choices=['last', 'mean', 'max', 'attention'], help='Pooling strategy')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--injection-ratio', type=float, default=0.5, help='Anomaly injection ratio')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--save-dir', type=str, default='results', help='Save directory')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = TrainingConfig(
        window_size=args.window_size,
        step=args.step,
        val_split=args.val_split,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        dropout=args.dropout,
        pool_strategy=args.pool,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        injection_ratio=args.injection_ratio,
        seed=args.seed,
        save_dir=args.save_dir,
        verbose=not args.quiet,
    )
    
    model, history, config = train(
        channel_id=args.channel,
        data_dir=args.data_dir,
        config=config,
    )
    
    return model, history


# ============================================================================
# Quick test
# ============================================================================

if __name__ == "__main__":
    # Check if running with arguments
    if len(sys.argv) > 1:
        main()
    else:
        # Quick test with defaults
        print("Running quick training test...")
        print("Use --help for CLI options\n")
        
        config = TrainingConfig(
            window_size=100,
            epochs=5,  # Quick test
            batch_size=32,
            d_model=32,  # Smaller for quick test
            n_layers=2,
            verbose=True,
        )
        
        # Try to train on first available channel
        try:
            loader = DataLoader(data_dir='../data')
            channels = loader.list_channels()
            if channels:
                model, history, _ = train(
                    channel_id=channels[0],
                    data_dir='../data',
                    config=config,
                )
                print("\n✓ Training test passed!")
            else:
                print("No channels found. Please check data directory.")
        except FileNotFoundError as e:
            print(f"Data not found: {e}")
            print("Please ensure data is in ../data/raw/train/ and ../data/raw/test/")
