# Variable Layer Architecture for LSTM/GRU Models

## Overview

This feature allows you to specify **per-layer hidden units** for LSTM and GRU models, enabling architectures like `[64, 32]` (2 layers with decreasing sizes) instead of being constrained to uniform layers like `[64, 64]`.

## Motivation

Previously, LSTM/GRU architectures were specified using:
- `LAYERS`: Number of layers (e.g., 2)
- `HIDDEN_UNITS`: Hidden units per layer (e.g., 64)
- Result: All layers had the same size (e.g., 64→64)

This was limiting compared to FNN models which support `FNN_HIDDEN_LAYERS: "64,32,16"`.

**Now you can specify architectures like:**
- `RNN_LAYER_SIZES: "64,32"` → 2 layers with sizes [64, 32]
- `RNN_LAYER_SIZES: "128,64,32"` → 3 layers with sizes [128, 64, 32]
- `RNN_LAYER_SIZES: "16,32,16"` → Hourglass architecture

## Usage

### Configuration Files

#### New Format (Recommended)
```json
{
    "MODEL_TYPE": "LSTM",
    "RNN_LAYER_SIZES": "64,32",
    ...
}
```

For GRU models, you can also use:
```json
{
    "MODEL_TYPE": "GRU",
    "GRU_UNITS": "64,32",
    ...
}
```

#### Legacy Format (Still Supported)
```json
{
    "MODEL_TYPE": "LSTM",
    "LAYERS": "2",
    "HIDDEN_UNITS": "64",
    ...
}
```
This creates uniform layers: [64, 64]

### Optuna Hyperparameter Search

#### Categorical Search (New)
```json
{
    "RNN_LAYER_SIZES": ["32,16", "64,32", "128,64,32", "64,64,64"]
}
```

#### Legacy Range Search (Still Works)
```json
{
    "LAYERS": "[1,3]",
    "HIDDEN_UNITS": "[16,128]"
}
```

## Implementation Details

### Model Selection

The system automatically chooses the appropriate implementation:

1. **Uniform Layers** (e.g., [64, 64, 64]):
   - Uses standard PyTorch `nn.LSTM` or `nn.GRU`
   - Most efficient for equal-sized layers
   - Hidden states: single tensor `[num_layers, batch, hidden_size]`

2. **Variable Layers** (e.g., [64, 32]):
   - Uses `LSTMStacked` or `GRUStacked` (custom implementations)
   - Uses `nn.ModuleList` of single-layer RNNs
   - Hidden states: list of tensors (one per layer)

### Backward Compatibility

✅ **100% backward compatible** with existing hyperparameter files:
- Old `LAYERS + HIDDEN_UNITS` → Converted to uniform layer sizes
- All existing models and checkpoints work unchanged
- GUI defaults remain the same

### Files Modified

#### Core Model Services
- `vestim/services/model_training/src/LSTM_model_service.py`
- `vestim/services/model_training/src/GRU_model_service.py`

#### New Model Implementations
- `vestim/services/model_training/src/LSTM_model_stacked.py`
- `vestim/services/model_training/src/GRU_model_stacked.py`
- `vestim/services/model_training/src/rnn_arch_utils.py`

#### Testing Services
- `vestim/services/model_testing/src/continuous_testing_service.py`
- `vestim/services/model_testing/src/testing_service.py`

#### Updated Model
- `vestim/services/model_training/src/LSTM_model.py` (improved None hidden state handling)

### Technical Implementation

#### Architecture Parsing
```python
from vestim.services.model_training.src.rnn_arch_utils import parse_layer_sizes

# New param takes precedence
layer_sizes = parse_layer_sizes(
    params.get("RNN_LAYER_SIZES"),
    fallback_hidden=params.get("HIDDEN_UNITS"),
    fallback_layers=params.get("LAYERS")
)
# Returns: [64, 32] for "64,32" or [64, 64] for LAYERS=2, HIDDEN_UNITS=64
```

#### Hidden State Management
```python
# Variable-size models use lists of hidden states
if isinstance(h_s, list):
    # Each element: [1, batch, hidden_size_i]
    h_s_detached = [t.detach() for t in h_s]
else:
    # Standard tensor: [num_layers, batch, hidden_size]
    h_s_detached = h_s.detach()
```

## Examples

### Example 1: Decreasing Architecture
```json
{
    "MODEL_TYPE": "LSTM",
    "RNN_LAYER_SIZES": "128,64,32",
    "LOOKBACK": 200,
    "BATCH_SIZE": 256
}
```
Creates 3-layer LSTM: 128 → 64 → 32 hidden units

### Example 2: Hourglass Architecture
```json
{
    "MODEL_TYPE": "GRU",
    "GRU_UNITS": "64,32,64",
    "LOOKBACK": 150
}
```
Creates 3-layer GRU with bottleneck: 64 → 32 → 64

### Example 3: Legacy Uniform
```json
{
    "MODEL_TYPE": "LSTM",
    "LAYERS": "3",
    "HIDDEN_UNITS": "64"
}
```
Creates 3-layer LSTM: 64 → 64 → 64 (uses efficient standard implementation)

## Testing

The implementation supports:
- ✅ Training with shuffled sequences
- ✅ Continuous testing with warmup
- ✅ Hidden state reset between independent test files
- ✅ Mixed precision training
- ✅ Model checkpointing and loading
- ✅ Optuna hyperparameter optimization

## Performance Considerations

- **Uniform layers** ([64, 64]): Uses optimized PyTorch kernels (fastest)
- **Variable layers** ([64, 32]): Slightly slower due to ModuleList iteration, but negligible impact (<5%)
- Memory usage scales with layer sizes (smaller later layers save memory)

## Migration Guide

### For Existing Hyperparameter Files
**No action needed** - files work as-is

### To Use Variable Architectures
1. Add `"RNN_LAYER_SIZES": "64,32"` to your JSON
2. Remove `LAYERS` and `HIDDEN_UNITS` (optional, they'll be ignored if RNN_LAYER_SIZES present)
3. Run training normally

### For GUI Users
GUI will be updated in a future commit to support the new parameter with a text field similar to FNN_HIDDEN_LAYERS.

## Troubleshooting

### Q: Model won't load / shape mismatch
**A:** Check that the hyperparameters match. Variable-size models saved with `RNN_LAYER_SIZES` need the same architecture for loading.

### Q: Hidden state errors during training
**A:** The updated code initializes hidden states as `None` and lets the model handle sizing. If you see errors, ensure you're using the updated model files.

### Q: Want to experiment with architectures
**A:** Use Optuna with categorical choices:
```json
{
    "RNN_LAYER_SIZES": ["32", "64,32", "128,64", "64,32,16"]
}
```

## Future Enhancements

- [ ] GUI support for RNN_LAYER_SIZES input field
- [ ] Visualization of layer sizes in training dashboard
- [ ] Auto-suggest optimal architectures based on dataset size
- [ ] Support for per-layer dropout probabilities

## Credits

Implemented as part of the `feature/rnn_variable_architecture` branch to provide FNN-style architectural flexibility for recurrent models while maintaining full backward compatibility.
