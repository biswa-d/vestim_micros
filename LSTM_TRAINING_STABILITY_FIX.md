# LSTM Training Stability Fix

## Problem Description

LSTM models were exhibiting erratic training behavior with sudden loss spikes during training, while FNN models trained stably on the same data. This is visible in the training plots where validation loss suddenly jumps from ~45 mV to ~1000 mV around epoch 120.

## Root Cause Analysis

### The Core Issue: Gradient Graph Accumulation

In PyTorch, when training RNN models (LSTM/GRU), hidden states carry computation graphs that link batches together. If these hidden states are not properly detached after each backward pass, the gradient computation graph accumulates across batches, leading to:

1. **Memory buildup** - Each batch adds to the computation graph
2. **Gradient instability** - Long chains amplify numerical errors
3. **Exploding gradients** - Even with gradient clipping at max_norm=0.5
4. **Training spikes** - Sudden jumps in loss when accumulated errors exceed thresholds

### Why FNN Doesn't Have This Problem

FNN (Feedforward Neural Networks) process each batch independently with no recurrent connections. Each forward pass creates a fresh computation graph that is destroyed after backward pass. There's no hidden state to accumulate gradients across batches.

### Previous Implementation Flaw

The code was resetting hidden states to `None` after each batch (for shuffled sequence training), but **the detachment was missing in the normal execution path**. Hidden states were only detached when errors were detected (invalid loss or gradients), meaning:

```python
# OLD CODE - only detached on errors
if not torch.isfinite(loss):
    if h_s is not None:
        h_s = h_s.detach()  # Only happens on error!
    continue

# Normal path - NO DETACHMENT before next iteration
total_train_loss.append(loss.item())
# ... next batch iteration with accumulated gradient graph
```

This meant that for epochs 1-119, gradients accumulated successfully, but by epoch 120+, the cumulative numerical errors and gradient chain length caused instability.

## Solution

Added explicit `.detach()` calls **after successful backward pass** but **before next batch iteration** in both training paths:

### Standard Precision Training Path
```python
# After optimizer.step()
if model_type in ["LSTM", "LSTM_EMA", "LSTM_LPF"]:
    if h_s is not None:
        h_s = h_s.detach()
    if h_c is not None:
        h_c = h_c.detach()
elif model_type == "GRU":
    if h_s is not None:
        h_s = h_s.detach()
```

### Mixed Precision Training Path
```python
# After scaler.step(optimizer) and scaler.update()
if model_type in ["LSTM", "LSTM_EMA", "LSTM_LPF"]:
    if h_s is not None:
        h_s = h_s.detach()
    if h_c is not None:
        h_c = h_c.detach()
elif model_type == "GRU":
    if h_s is not None:
        h_s = h_s.detach()
```

## Technical Details

### What `.detach()` Does

`tensor.detach()` creates a new tensor that shares the same data but is detached from the computation graph. This means:
- The tensor values (hidden states) are preserved for the next forward pass
- The gradient computation graph is severed, preventing backpropagation through it
- Memory is freed from the old computation graph
- Each batch starts with a fresh gradient graph

### Why This is Critical for RNNs

1. **Shuffled Training Data**: When training with shuffled sequences, batches are temporally disconnected. Carrying gradient graphs across disconnected sequences is incorrect.

2. **Gradient Clipping Limitation**: Even with aggressive gradient clipping (max_norm=0.5), the accumulated computation graph can still cause numerical instability.

3. **Memory Efficiency**: Detaching prevents memory buildup from dead computation graphs.

### Placement Timing

The detachment must happen:
- ✅ **After** `optimizer.step()` - weights are already updated
- ✅ **Before** appending to loss history - ensures clean state
- ✅ **Before** next batch iteration - prevents graph accumulation

## Expected Results

With this fix, LSTM training should:
- ✅ Show smooth, monotonic validation loss curves similar to FNN
- ✅ No sudden spikes or instability
- ✅ Lower memory usage during training
- ✅ More consistent convergence behavior
- ✅ Better exploit mode performance (since model weights remain stable)

## Testing Recommendations

1. **Re-run the same training configuration** that showed instability
2. **Monitor for**:
   - Smooth validation loss curves
   - No sudden jumps after epoch 100+
   - Consistent patience counter behavior
   - Stable learning rate reduction with ReduceLROnPlateau

3. **Compare FNN vs LSTM** on same data - should now show similar stability profiles

## Files Modified

- `vestim/services/model_training/src/training_task_service.py`
  - Added hidden state detachment after optimizer.step() (standard precision path)
  - Added hidden state detachment after scaler.update() (mixed precision path)
  - Both additions in the normal execution path, not just error handling

## References

- PyTorch documentation on [Truncated Backpropagation Through Time (TBPTT)](https://pytorch.org/docs/stable/notes/rnn.html)
- [Gradient accumulation and detachment in RNNs](https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426)
