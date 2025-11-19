# LSTM Training NaN/Inf Loss Debugging Guide

## Problem Description

Your LSTM training was experiencing loss going to `nan` around epoch 80-90, with validation loss stuck at 67.28 mV. This is a classic **gradient explosion** problem in recurrent neural networks.

## Root Causes Identified

### 1. **Too Aggressive Gradient Clipping** (FIXED)
- **Problem**: Gradient clipping was set to `max_norm=0.5`, which is too restrictive
- **Why it matters**: 
  - With lookback=400, gradients need to flow through 400 timesteps
  - Too small clipping can paradoxically CAUSE explosions by allowing some batches to slip through
  - Or it over-clips useful gradients, preventing learning entirely
- **Fix**: Increased to `max_norm=1.0` (standard for LSTMs)
- **Location**: `training_task_service.py` lines 162, 239

### 2. **Very Long Sequences (LOOKBACK=400)**
- **Problem**: Backpropagating through 400 timesteps compounds numerical errors
- **Why it matters**:
  - Each timestep multiplies gradient by hidden-to-hidden weights
  - 400 multiplications → exponential growth or decay
  - Even small weight instabilities become catastrophic
- **Solution Options**:
  - **Reduce lookback** to 100-200 (recommended for stability)
  - **Use Truncated BPTT** (backprop through time in chunks)
  - **Increase network capacity** (more hidden units distribute gradient load)

### 3. **Small Network Size (10 hidden units, 1 layer)**
- **Problem**: Each parameter carries huge responsibility
- **Why it matters**:
  - If any single parameter goes bad → entire network collapses
  - Not enough capacity to learn complex temporal patterns over 400 steps
  - Small networks are more sensitive to gradient noise
- **Recommendation**: Try 32-64 hidden units, or 2 layers of 16 units each

### 4. **Missing Weight Regularization**
- **Problem**: No WEIGHT_DECAY in hyperparameters
- **Why it matters**:
  - Weights can grow unbounded during training
  - Large weights → large gradients → explosion
  - L2 regularization naturally constrains weight magnitudes
- **Recommendation**: Add `"WEIGHT_DECAY": "0.0001"` to hyperparameters

## Fixes Implemented

### 1. Increased Gradient Clipping Threshold
```python
# Changed from max_norm=0.5 to max_norm=1.0
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 2. Extended Model Type Coverage
Now clips gradients for all RNN types:
- LSTM
- GRU  
- LSTM_EMA
- LSTM_LPF

### 3. Added NaN Detection After Parameter Updates
```python
# Check if model parameters became invalid AFTER optimizer.step()
params_are_finite = all(torch.isfinite(p).all() for p in model.parameters() if p.requires_grad)
if not params_are_finite:
    raise RuntimeError("Training failed: Model parameters became invalid (NaN/Inf)")
```

This catches the exact moment when parameters go bad and provides diagnostic info.

## Recommended Hyperparameter Changes

### Option 1: Safer Training (Recommended to Start)
```json
{
    "LOOKBACK": "200",           // Reduced from 400
    "HIDDEN_UNITS": "32",        // Increased from 10
    "LAYERS": "1",               // Keep at 1 for now
    "INITIAL_LR": "0.0001",      // Keep same
    "WEIGHT_DECAY": "0.0001",    // NEW - adds L2 regularization
    "DROPOUT_PROB": "0.2",       // Moderate dropout
    "LSTM_USE_LAYERNORM": true   // Helps stabilize training
}
```

### Option 2: More Capacity (If Option 1 Works)
```json
{
    "LOOKBACK": "300",           // Still reduced from 400
    "RNN_LAYER_SIZES": "64,32",  // 2 layers: 64 → 32
    "INITIAL_LR": "0.00005",     // Lower LR for bigger network
    "WEIGHT_DECAY": "0.0001",
    "LSTM_DROPOUT_PROB": "0.3",  // Higher dropout for regularization
    "LSTM_USE_LAYERNORM": true
}
```

### Option 3: If You MUST Use Lookback=400
```json
{
    "LOOKBACK": "400",           // Keep long sequences
    "RNN_LAYER_SIZES": "128,64", // Much bigger network needed
    "INITIAL_LR": "0.00003",     // Much lower LR
    "WEIGHT_DECAY": "0.0005",    // Stronger regularization
    "LSTM_DROPOUT_PROB": "0.4",
    "LSTM_USE_LAYERNORM": true,
    "USE_MIXED_PRECISION": true, // Better numerical stability
    "BATCH_SIZE": "128"          // Smaller batches = more stable gradients
}
```

## Why Your Specific Configuration Was Problematic

Looking at your training log:
- **Epoch 89**: Train RMS Error: `nan`, Val RMS Error: `67.28 mV`
- **Time Per Epoch**: ~1:46s (very fast)
- **Patience Counter**: 37-40 (validation not improving)

This pattern indicates:
1. Training loss exploded (NaN) due to gradient issues
2. Validation loss plateaued because model stopped learning meaningful patterns
3. The model was likely outputting near-constant predictions (around dataset mean)

The combination of:
- Lookback=400 (very long sequences)
- Hidden_units=10 (very small network)
- No weight decay (unbounded weight growth)
- Max_norm=0.5 (too restrictive clipping)

Created a **perfect storm** for gradient explosion.

## Testing Your Fixes

### Quick Test (Recommended)
1. Use Option 1 hyperparameters above
2. Train for 100 epochs
3. Monitor:
   - Loss should decrease smoothly (no NaN)
   - Validation error should improve
   - Training time per epoch should be similar

### Diagnostic Logging
The new safety checks will print:
```
CRITICAL ERROR: Epoch X, Batch Y: Model parameters became NaN/Inf after optimizer step!
  This suggests the learning rate or gradient magnitudes are too large for your sequence length.
  Current settings: LR=0.000100, Lookback=400
```

If you see this, try:
1. Reduce learning rate by 10x
2. Reduce lookback by 50%
3. Increase network size by 3x

## Additional Safeguards Already in Place

Your code already has good protections:
-  Xavier initialization for weights (prevents initial explosion)
-  Orthogonal initialization for hidden-to-hidden weights (prevents vanishing gradients)
-  Loss checking before backprop (skips bad batches)
-  Gradient norm checking after clipping (skips invalid gradients)
-  Mixed precision support (better numerical stability on GPU)
-  Hidden state detachment (prevents graph accumulation)

## Summary of Changes Made

**File**: `vestim/services/model_training/src/training_task_service.py`

1. **Line 162**: Increased gradient clipping from 0.5 → 1.0 (mixed precision path)
2. **Line 162**: Added LSTM_EMA and LSTM_LPF to gradient clipping
3. **Line 239**: Increased gradient clipping from 0.5 → 1.0 (standard precision path)
4. **Line 239**: Added LSTM_EMA and LSTM_LPF to gradient clipping
5. **Lines 180-186**: Added parameter NaN detection after mixed precision update
6. **Lines 256-262**: Added parameter NaN detection after standard precision update

## Next Steps

1. **Update your hyperparams.json** with Option 1 settings above
2. **Restart training** and monitor for NaN
3. **If stable**: Gradually increase lookback or decrease network size
4. **If still NaN**: Further reduce learning rate or lookback
5. **Track validation error**: Should see steady improvement now

## Long-Term Recommendations

1. **Implement learning rate warmup**: Start with very low LR for first few epochs
2. **Add early gradient monitoring**: Log gradient norms every batch initially
3. **Consider Transformer architecture**: Better than LSTM for very long sequences
4. **Use learning rate scheduling**: ReduceLROnPlateau or CosineAnnealing
5. **Add gradient noise**: Small random noise can prevent getting stuck in bad regions

## References

- Gradient clipping best practices: Pascanu et al. (2013) "On the difficulty of training RNNs"
- LSTM stability: Hochreiter & Schmidhuber (1997) original LSTM paper
- Weight initialization: Glorot & Bengio (2010) "Understanding the difficulty of training deep feedforward neural networks"
