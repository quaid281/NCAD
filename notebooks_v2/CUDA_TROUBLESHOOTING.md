# CUDA Error Mitigation

## Problem
During long training runs (especially with larger models or batch sizes), you may encounter:
```
torch.AcceleratorError: CUDA error: unknown error
```

## Root Causes
1. **GPU Memory Accumulation** - Memory not properly released between epochs
2. **GPU Overheating** - Extended compute causes thermal throttling
3. **Async CUDA Errors** - Errors reported late, making debugging difficult
4. **Memory Fragmentation** - Small allocations accumulate over time

## Mitigations Implemented

### 1. Periodic Cache Clearing
```python
# Every 50 batches during training
if n_batches % 50 == 0:
    torch.cuda.empty_cache()

# After each epoch
torch.cuda.empty_cache()
```

### 2. CUDA Synchronization
```python
# After validation to catch async errors early
torch.cuda.synchronize()
```

### 3. Graceful Error Recovery
```python
try:
    train_epoch(...)
except RuntimeError as e:
    if 'CUDA' in str(e):
        # Stop training but preserve best checkpoint
        torch.cuda.empty_cache()
        break
```

### 4. Conservative Training Settings
- **Batch size**: Start with 32-64 (not 128+)
- **Model size**: Use d_model=64 for channels with <10K samples
- **Early stopping**: Patience=10 prevents unnecessary long runs
- **Gradient clipping**: Prevents exploding gradients that can destabilize GPU

## Usage Recommendations

### For Short Experiments (< 20 epochs)
```bash
python train.py --channel D-3 --epochs 20 --batch-size 64
```

### For Long Training (50+ epochs)
```bash
# Reduce batch size to minimize memory pressure
python train.py --channel D-3 --epochs 50 --batch-size 32 --d-model 64
```

### If CUDA Errors Persist
1. **Restart Python kernel** - Clears any corrupted CUDA context
2. **Check GPU temperature** - Use `nvidia-smi` to monitor
3. **Reduce model size** - Use `--d-model 32 --n-layers 2`
4. **Use CPU** - Add `--device cpu` (slower but more stable)
5. **Update GPU drivers** - Ensure CUDA toolkit is up-to-date

## Monitoring GPU Health

```bash
# Watch GPU usage in real-time
nvidia-smi -l 1

# Check memory usage
nvidia-smi --query-gpu=memory.used,memory.free,temperature.gpu --format=csv -l 1
```

## Recovery Steps After CUDA Error

1. **Check for saved checkpoint**:
   ```bash
   ls results/D-3/best_model.pt
   ```

2. **Evaluate saved model**:
   ```bash
   python evaluate.py --channel D-3 --checkpoint results/D-3/best_model.pt
   ```

3. **Resume training from checkpoint** (if needed):
   - Currently not implemented, but could be added
   - Model was saved before error occurred

## Best Practices

✅ **DO**:
- Start with small batch sizes (32-64)
- Use early stopping (patience=10-15)
- Monitor GPU temperature during training
- Clear CUDA cache between runs
- Save checkpoints frequently

❌ **DON'T**:
- Run multiple training jobs on same GPU simultaneously
- Use batch_size > 128 on limited VRAM (< 8GB)
- Train for 100+ epochs without monitoring
- Ignore temperature warnings from `nvidia-smi`

## Architecture-Specific Notes

### Mamba/SSM Models
- More memory-intensive than standard RNNs due to state matrices
- Benefit from gradient checkpointing (not yet implemented)
- Consider reducing `d_state` from 16 to 8 for memory savings

### Current Model Sizes
- `d_model=32, n_layers=2`: ~20K parameters (lightweight)
- `d_model=64, n_layers=4`: ~130K parameters (standard)
- `d_model=128, n_layers=6`: ~1M parameters (heavy, may cause issues)

## Contact
If errors persist after trying these mitigations, consider:
1. Reducing model capacity further
2. Using gradient accumulation instead of large batches
3. Implementing mixed-precision training (FP16)
