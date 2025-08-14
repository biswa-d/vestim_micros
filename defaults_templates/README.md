# VEstim Hyperparameter Templates

This directory contains ready-to-use hyperparameter configuration templates for different model types and optimization methods.

## Template Types

### Grid Search Templates
- **`grid_search_lstm.json`** - LSTM model with exhaustive parameter search
- **`grid_search_gru.json`** - GRU model with exhaustive parameter search  
- **`grid_search_fnn.json`** - FNN model with exhaustive parameter search

### Optuna (Auto Search) Templates
- **`optuna_lstm.json`** - LSTM model with intelligent Bayesian optimization
- **`optuna_gru.json`** - GRU model with intelligent Bayesian optimization
- **`optuna_fnn.json`** - FNN model with intelligent Bayesian optimization

### Default Configuration
- **`hyperparams_last_used.json`** - Last successfully used configuration (automatically updated)

## Template Comparison

### Grid Search vs Optuna Format
- **Grid Search**: `"BATCH_SIZE": "128,256"` (comma-separated values)
- **Optuna**: `"BATCH_SIZE": "[128,512]"` (min,max boundaries for optimization)

### Parameter Counts
- **Grid Search**: 2-3 values per parameter (manageable combinations)
- **Optuna**: Continuous ranges (infinite possibilities within bounds)

## Usage Instructions

### 1. Loading Templates
1. Open VEstim
2. Go to Hyperparameter Configuration
3. Click "Load Parameters from File"
4. Select appropriate template from this directory

### 2. Choosing the Right Template

**Use Grid Search when:**
- You want to test specific parameter combinations
- You have limited computational resources
- You prefer predictable, exhaustive search
- You want to compare exact parameter sets

**Use Optuna when:**
- You want intelligent parameter optimization
- You have more computational resources
- You want to find optimal parameters efficiently
- You're exploring new parameter ranges

### 3. Model Type Selection
- **LSTM**: Good for long sequences, complex patterns
- **GRU**: Faster than LSTM, simpler architecture
- **FNN**: Fast training, works well for non-sequential features

## Customization Tips

### For Grid Search Templates:
1. Add/remove values from comma-separated lists
2. Keep combinations reasonable (2x2x2 = 8 combinations max recommended)
3. Example: `"BATCH_SIZE": "64,128,256"` → 3 values to try

### For Optuna Templates:
1. Adjust [min,max] ranges based on your needs
2. Wider ranges = more exploration, longer optimization
3. Example: `"BATCH_SIZE": "[32,512]"` → search between 32 and 512

## Template Modifications

### Conservative Approach (Faster):
```json
{
    "BATCH_SIZE": "128,256",        // Grid: 2 values
    "BATCH_SIZE": "[128,256]",      // Optuna: narrow range
    "MAX_EPOCHS": "50,100",         // Grid: 2 values  
    "MAX_EPOCHS": "[50,100]"        // Optuna: limited epochs
}
```

### Aggressive Approach (More thorough):
```json
{
    "BATCH_SIZE": "64,128,256",     // Grid: 3 values
    "BATCH_SIZE": "[32,512]",       // Optuna: wide range
    "MAX_EPOCHS": "100,200,300",    // Grid: 3 values
    "MAX_EPOCHS": "[100,500]"       // Optuna: long training
}
```

## Performance Settings

All templates include optimized performance settings:
- **Mixed Precision**: Faster training on modern GPUs
- **Multi-threading**: Efficient data loading
- **GPU Memory**: Optimized batch processing

## Results

After successful training, your configuration will be saved as the new `hyperparams_last_used.json`, making it easy to repeat successful experiments or use as a starting point for future optimizations.
