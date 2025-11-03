# RNN_LAYER_SIZES Parameter Guide

## Overview
The `RNN_LAYER_SIZES` parameter provides an FNN-style way to specify LSTM/GRU architectures using comma-separated layer sizes.

## Usage

### Single Architecture
```json
{
  "RNN_LAYER_SIZES": "64,32"
}
```
This creates a 2-layer LSTM with 64 units in layer 1 and 32 units in layer 2.

### Multiple Architectures (Grid Search)
```json
{
  "RNN_LAYER_SIZES": "64,32;128,64;32,32"
}
```
This tries three different architectures:
- Architecture 1: 64 → 32 (2 layers)
- Architecture 2: 128 → 64 (2 layers)  
- Architecture 3: 32 → 32 (2 layers, uniform)

### GUI Entry
In the hyperparameter GUI:
- **LSTM models**: Enter in "LSTM Layer Sizes" field: `64,32`
- **GRU models**: Enter in "GRU Layer Sizes" field: `128,64`
- **Multiple configs**: Use semicolons: `64,32;128,64;256,128`

## Backward Compatibility

Old JSON files still work:
```json
{
  "LAYERS": 2,
  "HIDDEN_UNITS": 64
}
```
This is automatically converted to `RNN_LAYER_SIZES = "64,64"` internally.

## Parameter Precedence

The system checks parameters in this order:
1. `RNN_LAYER_SIZES` (new, highest priority)
2. `LSTM_UNITS` or `GRU_UNITS` (aliases)
3. `LAYERS` + `HIDDEN_UNITS` (legacy, fallback)

## Implementation Details

- **Uniform layers** (e.g., `"64,64"`): Uses standard `nn.LSTM` with all layers same size
- **Currently**: Variable layers (e.g., `"64,32"`) use the first value for all layers
- **Future**: Can be enhanced to support truly variable layer sizes with custom implementation

## Examples

### Training Config
```json
{
  "MODEL_TYPE": "LSTM",
  "RNN_LAYER_SIZES": "128,64,32",
  "BATCH_SIZE": 32,
  "INITIAL_LR": 0.001
}
```

### Optuna Search (Not Yet Supported)
For Optuna, currently use legacy params:
```json
{
  "LAYERS": "[1,3]",
  "HIDDEN_UNITS": "[32,128]"
}
```

### Grid Search with Multiple Architectures
```json
{
  "MODEL_TYPE": "LSTM",
  "RNN_LAYER_SIZES": "32,16;64,32;128,64;64,64",
  "BATCH_SIZE": "16,32,64",
  "INITIAL_LR": "0.001,0.0001"
}
```

## File Naming

Models are saved with descriptive folder names:
- **New format**: `LSTM_ARCH_64_32` (uses RNN_LAYER_SIZES)
- **Legacy format**: `LSTM_L2_HU64` (uses LAYERS + HIDDEN_UNITS)

## Migration

No migration needed! Your existing configs continue to work. To use the new format:

1. Open hyperparameter GUI
2. Select LSTM or GRU model
3. Enter layer sizes in the single "Layer Sizes" field: `64,32`
4. Save and run

That's it!
