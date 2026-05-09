"""
Evaluation Script for Mamba Anomaly Classifier (MAC)

Evaluates trained models on test data with:
- Point-level and window-level metrics
- Visualization of predictions vs ground truth
- Score distribution analysis
- Threshold sensitivity analysis
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc,
    confusion_matrix, classification_report
)

# Local imports
from models.mamba_encoder import MambaAnomalyClassifier
from utils.data_loader import DataLoader


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    channel: str
    checkpoint_path: str
    data_dir: str = "../data"
    labels_path: Optional[str] = "../data/processed/labeled_anomalies.csv"
    window_size: int = 256
    stride: int = 1
    batch_size: int = 64
    threshold: float = 0.5
    device: str = "auto"
    output_dir: str = "results"
    save_plots: bool = True
    verbose: bool = True


@dataclass 
class EvaluationResults:
    """Container for evaluation results."""
    # Raw outputs
    probabilities: np.ndarray
    predictions: np.ndarray
    labels: Optional[np.ndarray]
    
    # Metrics (if labels available)
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    accuracy: Optional[float] = None
    
    # Confusion matrix components
    tp: Optional[int] = None
    tn: Optional[int] = None
    fp: Optional[int] = None
    fn: Optional[int] = None
    
    # Additional info
    threshold: float = 0.5
    num_windows: int = 0
    num_anomalies_predicted: int = 0
    num_anomalies_true: Optional[int] = None


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[MambaAnomalyClassifier, dict]:
    """Load trained model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model config - check both 'model_config' and 'config' keys
    model_config = checkpoint.get('model_config', checkpoint.get('config', {}))
    
    # Create model with saved config
    model = MambaAnomalyClassifier(
        input_dim=model_config.get('input_dim', 1),
        d_model=model_config.get('d_model', 64),
        d_state=model_config.get('d_state', 16),
        d_conv=model_config.get('d_conv', 4),
        expand=model_config.get('expand', 2),
        n_layers=model_config.get('n_layers', 4),
        dropout=model_config.get('dropout', 0.1),
        pool_strategy=model_config.get('pool_strategy', 'last')
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get training info
    training_info = {
        'epoch': checkpoint.get('epoch', 'unknown'),
        'best_val_loss': checkpoint.get('best_val_loss', checkpoint.get('val_loss', 'unknown')),
        'config': checkpoint.get('config', {}),
        'model_config': model_config
    }
    
    print(f"  Model loaded from epoch {training_info['epoch']}")
    print(f"  Best validation loss: {training_info['best_val_loss']}")
    
    return model, training_info


def load_ground_truth(labels_path: str, channel: str, test_length: int) -> Optional[np.ndarray]:
    """Load ground truth labels for a channel."""
    import pandas as pd
    
    if not os.path.exists(labels_path):
        print(f"Warning: Labels file not found at {labels_path}")
        return None
    
    try:
        df = pd.read_csv(labels_path)
        channel_data = df[df['chan_id'] == channel]
        
        if channel_data.empty:
            print(f"Warning: No labels found for channel {channel}")
            return None
        
        # Parse anomaly sequences
        anomaly_sequences_str = channel_data['anomaly_sequences'].iloc[0]
        if isinstance(anomaly_sequences_str, str):
            anomaly_sequences = eval(anomaly_sequences_str)
        else:
            anomaly_sequences = []
        
        # Create binary labels
        num_values = channel_data['num_values'].iloc[0]
        binary_labels = np.zeros(num_values, dtype=np.float32)
        
        for anomaly in anomaly_sequences:
            if isinstance(anomaly, (list, tuple)) and len(anomaly) == 2:
                start, end = anomaly
                binary_labels[start:end] = 1.0
            elif isinstance(anomaly, int):
                binary_labels[anomaly] = 1.0
        
        # Adjust length if needed
        if len(binary_labels) != test_length:
            if len(binary_labels) < test_length:
                binary_labels = np.pad(binary_labels, (0, test_length - len(binary_labels)))
            else:
                binary_labels = binary_labels[:test_length]
        
        num_anomalies = int(np.sum(binary_labels))
        print(f"  Ground truth loaded: {num_anomalies} anomalous points ({100*num_anomalies/len(binary_labels):.2f}%)")
        
        return binary_labels
        
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return None


def create_window_labels(point_labels: np.ndarray, window_size: int, stride: int,
                         anomaly_threshold: float = 0.0) -> np.ndarray:
    """Convert point-level labels to window-level labels.
    
    A window is labeled anomalous if the fraction of anomalous points
    exceeds the anomaly_threshold.
    """
    n_windows = (len(point_labels) - window_size) // stride + 1
    window_labels = np.zeros(n_windows, dtype=np.float32)
    
    for i in range(n_windows):
        start = i * stride
        end = start + window_size
        window_anomaly_ratio = np.mean(point_labels[start:end])
        window_labels[i] = 1.0 if window_anomaly_ratio > anomaly_threshold else 0.0
    
    return window_labels


def evaluate_model(
    model: MambaAnomalyClassifier,
    test_windows: np.ndarray,
    window_labels: Optional[np.ndarray],
    config: EvaluationConfig,
    device: torch.device
) -> EvaluationResults:
    """Run evaluation on test data."""
    print(f"\nEvaluating on {len(test_windows)} windows...")
    
    model.eval()
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(test_windows), config.batch_size):
            batch = test_windows[i:i + config.batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            
            # Get model outputs - model returns logits only by default
            logits = model(batch_tensor)
            probs = torch.sigmoid(logits).squeeze(-1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.append((probs >= config.threshold).cpu().numpy())
    
    # Concatenate results
    probabilities = np.concatenate(all_probs)
    predictions = (probabilities >= config.threshold).astype(np.float32)
    
    # Create results object
    results = EvaluationResults(
        probabilities=probabilities,
        predictions=predictions,
        labels=window_labels,
        threshold=config.threshold,
        num_windows=len(test_windows),
        num_anomalies_predicted=int(np.sum(predictions))
    )
    
    # Calculate metrics if labels available
    if window_labels is not None:
        y_true = window_labels.astype(int)
        y_pred = predictions.astype(int)
        
        results.precision = precision_score(y_true, y_pred, zero_division=0)
        results.recall = recall_score(y_true, y_pred, zero_division=0)
        results.f1 = f1_score(y_true, y_pred, zero_division=0)
        results.accuracy = np.mean(y_true == y_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            results.tn, results.fp, results.fn, results.tp = cm.ravel()
        
        results.num_anomalies_true = int(np.sum(window_labels))
    
    return results


def find_optimal_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """Find optimal threshold for a given metric."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = 0.0
    
    for thresh in thresholds:
        preds = (probabilities >= thresh).astype(int)
        
        if metric == 'f1':
            score = f1_score(labels, preds, zero_division=0)
        elif metric == 'precision':
            score = precision_score(labels, preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(labels, preds, zero_division=0)
        else:
            score = f1_score(labels, preds, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def plot_results(
    results: EvaluationResults,
    test_signal: np.ndarray,
    window_size: int,
    stride: int,
    channel: str,
    output_dir: str,
    training_info: dict
):
    """Generate visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Main detection plot - 2 panels
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    
    # Create time axis for windows
    n_windows = len(results.probabilities)
    window_centers = np.array([i * stride + window_size // 2 for i in range(n_windows)])
    
    # Panel 1: Signal with anomaly regions
    ax1 = axes[0]
    ax1.plot(test_signal, 'b-', alpha=0.7, linewidth=0.5, label='Signal')
    ax1.set_ylabel('Signal Value', fontsize=11)
    
    # Build title with metrics if available
    if results.labels is not None:
        title = f'Mamba Anomaly Classifier - Channel {channel} | P={results.precision:.3f} R={results.recall:.3f} F1={results.f1:.3f}'
    else:
        title = f'Mamba Anomaly Classifier - Channel {channel}'
    ax1.set_title(title, fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Show ground truth regions (if available)
    if results.labels is not None:
        gt_mask = np.zeros(len(test_signal), dtype=bool)
        for i, label in enumerate(results.labels):
            if label > 0.5:
                start = i * stride
                end = min(start + window_size, len(test_signal))
                gt_mask[start:end] = True
        
        # Create spans for ground truth
        in_gt = False
        start_idx = 0
        for i, is_gt in enumerate(gt_mask):
            if is_gt and not in_gt:
                start_idx = i
                in_gt = True
            elif not is_gt and in_gt:
                ax1.axvspan(start_idx, i, alpha=0.2, color='green', label='Ground Truth' if start_idx == np.where(gt_mask)[0][0] else '')
                in_gt = False
        if in_gt:
            ax1.axvspan(start_idx, len(gt_mask), alpha=0.2, color='green')
    
    # Show predicted anomalies on signal
    if results.num_anomalies_predicted > 0:
        pred_mask = np.zeros(len(test_signal), dtype=bool)
        for i, pred in enumerate(results.predictions):
            if pred > 0.5:
                start = i * stride
                end = min(start + window_size, len(test_signal))
                pred_mask[start:end] = True
        
        # Create spans for predictions
        in_pred = False
        start_idx = 0
        first_pred = True
        for i, is_pred in enumerate(pred_mask):
            if is_pred and not in_pred:
                start_idx = i
                in_pred = True
            elif not is_pred and in_pred:
                ax1.axvspan(start_idx, i, alpha=0.3, color='red', label='Predicted' if first_pred else '')
                first_pred = False
                in_pred = False
        if in_pred:
            ax1.axvspan(start_idx, len(pred_mask), alpha=0.3, color='red')
    
    # Add legend with metrics
    if results.labels is not None:
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.2, label='Ground Truth'),
            Patch(facecolor='red', alpha=0.3, label='Predicted'),
            Patch(facecolor='green', edgecolor='green', label=f'TP: {results.tp}'),
            Patch(facecolor='red', edgecolor='red', label=f'FP: {results.fp}'),
            Patch(facecolor='orange', edgecolor='orange', label=f'FN: {results.fn}')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', ncol=5, fontsize=9)
    else:
        ax1.legend(loc='upper left')
    
    # Panel 2: Anomaly probability
    ax2 = axes[1]
    ax2.plot(window_centers, results.probabilities, 'b-', alpha=0.7, linewidth=0.8, label='Anomaly Score')
    ax2.axhline(y=results.threshold, color='r', linestyle='--', linewidth=1.5, label=f'Threshold ({results.threshold:.2f})')
    ax2.fill_between(window_centers, 0, results.probabilities, alpha=0.3, color='blue')
    ax2.set_ylabel('Anomaly Probability', fontsize=11)
    ax2.set_xlabel('Time Index', fontsize=11)
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{channel}_detection_results.png'), dpi=150, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, f'{channel}_detection_results.svg'), bbox_inches='tight')
    plt.close()
    
    # 2. Score distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1 = axes[0]
    ax1.hist(results.probabilities, bins=50, alpha=0.7, edgecolor='black')
    ax1.axvline(x=results.threshold, color='r', linestyle='--', label=f'Threshold ({results.threshold:.2f})')
    ax1.set_xlabel('Anomaly Probability')
    ax1.set_ylabel('Count')
    ax1.set_title('Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # If labels available, show distribution by class
    if results.labels is not None:
        ax2 = axes[1]
        normal_probs = results.probabilities[results.labels == 0]
        anomaly_probs = results.probabilities[results.labels == 1]
        
        if len(normal_probs) > 0:
            ax2.hist(normal_probs, bins=30, alpha=0.5, label='Normal', color='green', edgecolor='black')
        if len(anomaly_probs) > 0:
            ax2.hist(anomaly_probs, bins=30, alpha=0.5, label='Anomaly', color='red', edgecolor='black')
        
        ax2.axvline(x=results.threshold, color='black', linestyle='--', label=f'Threshold')
        ax2.set_xlabel('Anomaly Probability')
        ax2.set_ylabel('Count')
        ax2.set_title('Score Distribution by Class')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No labels for class breakdown', 
                    transform=axes[1].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{channel}_score_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. PR and ROC curves (if labels available)
    if results.labels is not None and results.num_anomalies_true > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Precision-Recall curve
        precision, recall, pr_thresholds = precision_recall_curve(results.labels, results.probabilities)
        pr_auc = auc(recall, precision)
        
        ax1 = axes[0]
        ax1.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC={pr_auc:.3f})')
        ax1.scatter([results.recall], [results.precision], c='red', s=100, zorder=5, 
                   label=f'Current (t={results.threshold:.2f})')
        ax1.set_xlabel('Recall')
        ax1.set_ylabel('Precision')
        ax1.set_title('Precision-Recall Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(results.labels, results.probabilities)
        roc_auc = auc(fpr, tpr)
        
        ax2 = axes[1]
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC={roc_auc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{channel}_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def print_results(results: EvaluationResults, channel: str, training_info: dict):
    """Print evaluation results summary."""
    print("\n" + "="*60)
    print(f"EVALUATION RESULTS - Channel {channel}")
    print("="*60)
    
    print(f"\nModel Info:")
    print(f"  Trained epochs: {training_info.get('epoch', 'N/A')}")
    print(f"  Best val loss:  {training_info.get('best_val_loss', 'N/A')}")
    
    print(f"\nPrediction Summary:")
    print(f"  Total windows:        {results.num_windows:,}")
    print(f"  Predicted anomalies:  {results.num_anomalies_predicted:,} ({100*results.num_anomalies_predicted/results.num_windows:.2f}%)")
    print(f"  Threshold:            {results.threshold:.3f}")
    
    if results.labels is not None:
        print(f"  True anomalies:       {results.num_anomalies_true:,} ({100*results.num_anomalies_true/results.num_windows:.2f}%)")
        
        print(f"\nClassification Metrics:")
        print(f"  Precision:  {results.precision:.4f}")
        print(f"  Recall:     {results.recall:.4f}")
        print(f"  F1 Score:   {results.f1:.4f}")
        print(f"  Accuracy:   {results.accuracy:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TP: {results.tp:,}  |  FP: {results.fp:,}")
        print(f"  FN: {results.fn:,}  |  TN: {results.tn:,}")
    else:
        print(f"\n  (No ground truth labels available for metrics)")
    
    print("="*60)


def save_results(results: EvaluationResults, channel: str, output_dir: str, training_info: dict):
    """Save evaluation results to JSON."""
    results_dict = {
        'channel': channel,
        'threshold': results.threshold,
        'num_windows': results.num_windows,
        'num_anomalies_predicted': results.num_anomalies_predicted,
        'num_anomalies_true': results.num_anomalies_true,
        'precision': results.precision,
        'recall': results.recall,
        'f1': results.f1,
        'accuracy': results.accuracy,
        'tp': results.tp,
        'tn': results.tn,
        'fp': results.fp,
        'fn': results.fn,
        'model_epoch': training_info.get('epoch'),
        'model_val_loss': training_info.get('best_val_loss'),
        'model_config': training_info.get('model_config', {})
    }
    
    output_path = os.path.join(output_dir, f'{channel}_evaluation.json')
    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    
    # Save predictions as numpy
    np.save(os.path.join(output_dir, f'{channel}_probabilities.npy'), results.probabilities)
    np.save(os.path.join(output_dir, f'{channel}_predictions.npy'), results.predictions)
    
    print(f"Results saved to {output_path}")


def evaluate(config: EvaluationConfig) -> EvaluationResults:
    """Main evaluation function."""
    # Setup device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    print(f"Using device: {device}")
    
    # Load model
    model, training_info = load_model(config.checkpoint_path, device)
    
    # Load test data
    print(f"\nLoading test data for channel {config.channel}...")
    data_loader = DataLoader(config.data_dir)
    channel_data = data_loader.load_channel(config.channel)
    
    if channel_data is None:
        raise ValueError(f"Failed to load test data for channel {config.channel}")
    
    print(f"  Test signal length: {len(channel_data.test_raw):,}")
    print(f"  Signal stats: mean={channel_data.norm_stats.mean:.4f}, std={channel_data.norm_stats.std:.4f}")
    
    # Create windows from normalized test data
    test_windows = data_loader.create_windows(
        channel_data.test_normalized,
        window_size=config.window_size,
        step=config.stride
    )
    print(f"  Created {len(test_windows):,} test windows")
    
    # Use labels directly from channel_data
    point_labels = channel_data.labels
    window_labels = None
    
    if point_labels is not None:
        # Convert to window labels
        window_labels = create_window_labels(
            point_labels, 
            config.window_size, 
            config.stride,  # step between windows
            anomaly_threshold=0.0  # Any anomaly in window = anomalous window
        )
        print(f"  Window labels: {int(np.sum(window_labels)):,} anomalous windows")
    
    # Run evaluation
    results = evaluate_model(model, test_windows, window_labels, config, device)
    
    # Find optimal threshold if labels available
    if window_labels is not None:
        opt_threshold, opt_f1 = find_optimal_threshold(results.probabilities, window_labels, 'f1')
        print(f"\nOptimal threshold for F1: {opt_threshold:.3f} (F1={opt_f1:.4f})")
        
        if opt_threshold != config.threshold:
            print(f"  (Current threshold: {config.threshold:.3f}, F1={results.f1:.4f})")
    
    # Output results
    output_dir = os.path.join(config.output_dir, config.channel)
    os.makedirs(output_dir, exist_ok=True)
    
    print_results(results, config.channel, training_info)
    save_results(results, config.channel, output_dir, training_info)
    
    if config.save_plots:
        plot_results(
            results, 
            channel_data.test_normalized,
            config.window_size,
            config.stride,
            config.channel,
            output_dir,
            training_info
        )
    
    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Evaluate Mamba Anomaly Classifier',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--channel', '-c', type=str, required=True,
                        help='Channel ID to evaluate (e.g., A-1, D-3)')
    parser.add_argument('--checkpoint', '-m', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Data arguments
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Directory containing raw/ and processed/ subdirectories')
    parser.add_argument('--labels-path', type=str, default='../data/processed/labeled_anomalies.csv',
                        help='Path to labeled anomalies CSV')
    
    # Model arguments
    parser.add_argument('--window-size', '-w', type=int, default=256,
                        help='Window size for evaluation')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for sliding windows')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                        help='Classification threshold')
    
    # Output arguments
    parser.add_argument('--output-dir', '-o', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable plot generation')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Create config
    config = EvaluationConfig(
        channel=args.channel,
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        labels_path=args.labels_path if os.path.exists(args.labels_path) else None,
        window_size=args.window_size,
        stride=args.stride,
        threshold=args.threshold,
        output_dir=args.output_dir,
        save_plots=not args.no_plots,
        device=args.device
    )
    
    # Run evaluation
    results = evaluate(config)
    
    return 0 if results is not None else 1


if __name__ == "__main__":
    sys.exit(main())
