# VEstim Optuna Integration Implementation

## Overview

This implementation adds Optuna-based automatic hyperparameter optimization to VEstim, providing an intelligent alternative to exhaustive grid search.

## New Workflow

### Previous Flow
```
Hyperparameter GUI → Training Setup GUI
```

### New Flow
```
Hyperparameter GUI → Search Method Selection → 
├── Auto Search: Optuna Optimization GUI → Training Setup GUI (with optimized configs)
└── Grid Search: Training Setup GUI (exhaustive grid search)
```

## Key Features

### 1. Search Method Selection
- **Auto Search (Optuna)**: Uses Optuna's TPE sampler for intelligent hyperparameter optimization
- **Exhaustive Grid Search**: Traditional approach with all parameter combinations

### 2. Optuna Optimization GUI
- **Configuration Tab**: Set optimization parameters (trials, ranges)
- **Optimization Tab**: Real-time progress tracking and trial monitoring
- **Results Tab**: View and export best configurations

### 3. Flexible Parameter Handling
- Supports both single parameter sets (grid search) and multiple optimized configurations (Optuna)
- Automatic validation and type conversion
- Model-specific parameter suggestions

## Implementation Details

### Modified Files

#### 1. `hyper_param_gui_qt.py`
- **Added**: Search method selection buttons
- **Added**: `proceed_to_auto_search()` and `proceed_to_grid_search()` methods
- **Added**: `_collect_and_validate_params()` for shared parameter validation
- **Modified**: Button layout to include Auto Search and Grid Search options

#### 2. `optuna_optimization_gui_qt.py` (New)
- **Complete Optuna integration with tabbed interface**
- **Background optimization thread** with progress tracking
- **Parameter range configuration** based on model type
- **Real-time trial monitoring** and results display
- **Export functionality** for optimization results

#### 3. `training_setup_gui_qt.py`
- **Enhanced**: Constructor to handle both single and multiple parameter configurations
- **Updated**: Window titles and descriptions to reflect search method
- **Backward compatible**: Works with existing grid search workflow

#### 4. `requirements.txt`
- **Added**: `optuna==3.6.1` dependency

### Parameter Optimization Ranges

The implementation supports optimization ranges for:
- **Hidden Units**: Configurable range (default: 5-100)
- **Layers**: Configurable range (default: 1-3)
- **Learning Rate**: Log-scale range (default: 1e-5 to 1e-2)
- **Batch Size**: Categorical choices (default: [32, 64, 128, 256, 512])
- **Lookback Window**: For sequence models (default: 100-800)
- **Dropout Probability**: For FNN models (default: 0.0-0.5)

### Optimization Process

1. **Configuration**: User sets number of trials and parameter ranges
2. **Optimization**: Optuna runs trials using TPE sampler
3. **Evaluation**: Each trial evaluates parameter combination (placeholder for actual training)
4. **Selection**: Best N configurations selected based on objective value
5. **Training**: Selected configurations passed to Training Setup GUI

## User Experience

### Auto Search Flow
1. Configure hyperparameters in Hyperparameter GUI
2. Click "Auto Search (Optuna)"
3. Set optimization parameters (trials, ranges)
4. Start optimization and monitor progress
5. Review results and proceed with best configurations

### Grid Search Flow (Unchanged)
1. Configure hyperparameters in Hyperparameter GUI
2. Click "Exhaustive Grid Search"
3. Proceed directly to Training Setup GUI

## Benefits

### For Users
- **Intelligent Search**: Optuna finds good configurations faster than grid search
- **Time Savings**: Fewer trials needed compared to exhaustive search
- **Flexibility**: Choose between automatic and manual approaches
- **Progress Tracking**: Real-time monitoring of optimization progress

### For Developers
- **Modular Design**: Clean separation between optimization and training
- **Backward Compatibility**: Existing grid search workflow unchanged
- **Extensible**: Easy to add new optimization algorithms or parameters
- **Error Handling**: Robust error handling and user feedback

## Installation

```bash
pip install optuna==3.6.1
```

## Testing

Run the test script to verify the implementation:
```bash
python test_optuna_integration.py
```

## Future Enhancements

### Planned Features
1. **Actual Training Evaluation**: Replace placeholder with real model training
2. **Multi-objective Optimization**: Optimize for multiple metrics (accuracy, speed, etc.)
3. **Pruning**: Early stopping of unpromising trials
4. **Advanced Samplers**: Support for other Optuna samplers (CMA-ES, GP, etc.)
5. **Hyperparameter Importance**: Analysis of which parameters matter most
6. **Cross-validation**: More robust evaluation metrics

### Advanced Options
- **Study Persistence**: Save/load optimization studies
- **Distributed Optimization**: Run trials across multiple machines
- **Custom Objective Functions**: User-defined optimization targets
- **Visualization**: Built-in optimization history plots

## Architecture Notes

### Thread Safety
- Optimization runs in separate thread to avoid GUI freezing
- Progress signals used for thread-safe communication
- Proper cleanup on optimization stop/cancel

### Error Handling
- Graceful degradation when Optuna not installed
- Validation of parameter ranges and types
- User-friendly error messages

### Memory Management
- Efficient handling of multiple parameter configurations
- Proper cleanup of optimization threads and resources

This implementation provides a solid foundation for intelligent hyperparameter optimization while maintaining the flexibility and robustness of the existing VEstim system.
