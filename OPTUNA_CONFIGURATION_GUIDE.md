# VEstim Optuna Configuration Guide

## Overview
This guide explains how to configure hyperparameters for Optuna-based automatic optimization in VEstim.

## Format Requirements

### Optuna Format vs Grid Search
- **Grid Search**: `"BATCH_SIZE": "100,200,400"` (comma-separated)
- **Optuna**: `"BATCH_SIZE": "[100,400]"` (min,max boundaries)

### Parameter Categories

#### 1. ALWAYS OPTIMIZED (Must use [min,max] format)
```json
{
    "BATCH_SIZE": "[32,512]",
    "MAX_EPOCHS": "[10,200]", 
    "INITIAL_LR": "[1e-5,1e-2]"
}
```

#### 2. MODEL-SPECIFIC OPTIMIZED PARAMETERS

**LSTM Models:**
```json
{
    "LAYERS": "[1,4]",
    "HIDDEN_UNITS": "[16,256]",
    "LOOKBACK": "[100,800]"
}
```

**GRU Models:**
```json
{
    "GRU_LAYERS": "[1,4]",
    "GRU_HIDDEN_UNITS": "[16,256]", 
    "LOOKBACK": "[100,800]"
}
```

**FNN Models:**
```json
{
    "FNN_HIDDEN_LAYERS": "[[32,64,128],[128,256,512]]",
    "FNN_DROPOUT_PROB": "[0.1,0.6]"
}
```

#### 3. SCHEDULER-SPECIFIC PARAMETERS

**StepLR:**
```json
{
    "LR_PARAM": "[0.1,0.9]",
    "LR_PERIOD": "[5,25]"
}
```

**ReduceLROnPlateau:**
```json
{
    "PLATEAU_FACTOR": "[0.1,0.8]",
    "PLATEAU_PATIENCE": "[5,20]"
}
```

**CosineAnnealingWarmRestarts:**
```json
{
    "COSINE_T0": "[5,30]",
    "COSINE_T_MULT": "[1,3]", 
    "COSINE_ETA_MIN": "[1e-8,1e-4]"
}
```

#### 4. FIXED PARAMETERS (Never optimized)
```json
{
    "DEVICE_SELECTION": "cuda:0",
    "FEATURE_COLUMNS": ["Power", "Battery_Temp_degC", "SOC"],
    "TARGET_COLUMN": "Voltage",
    "MODEL_TYPE": "LSTM",
    "OPTIMIZER_TYPE": "Adam",
    "SCHEDULER_TYPE": "StepLR",
    "TRAINING_METHOD": "Sequence-to-Sequence"
}
```

#### 5. FLEXIBLE PARAMETERS (Can be fixed or optimized)
```json
{
    "VALID_PATIENCE": "15",           // Fixed value
    "VALID_PATIENCE": "[10,25]",      // Or optimized range
    "REPETITIONS": 1                  // Usually fixed
}
```

## Parameter Ranges by Data Type

### Integer Parameters
- **Small ranges**: LAYERS `[1,4]`, REPETITIONS `[1,3]`
- **Medium ranges**: HIDDEN_UNITS `[16,256]`, LOOKBACK `[100,800]`
- **Large ranges**: BATCH_SIZE `[32,1024]`, MAX_EPOCHS `[10,500]`

### Float Parameters (Linear scale)
- **Probabilities**: FNN_DROPOUT_PROB `[0.0,0.8]`
- **Factors**: PLATEAU_FACTOR `[0.1,0.9]`, LR_PARAM `[0.1,0.9]`

### Float Parameters (Log scale)
- **Learning rates**: INITIAL_LR `[1e-6,1e-1]`
- **Minimum values**: COSINE_ETA_MIN `[1e-8,1e-4]`

## Usage Instructions

### 1. Loading Default Templates
Use the "Load Parameters from File" button in the Hyperparameter GUI to load:
- `defaults/optuna_hyperparams_lstm.json` - For LSTM models
- `defaults/optuna_hyperparams_gru.json` - For GRU models  
- `defaults/optuna_hyperparams_fnn.json` - For FNN models

### 2. Customizing Ranges
Adjust the [min,max] values based on your needs:
- **Narrow search**: `[50,150]` for focused optimization
- **Wide search**: `[10,500]` for exploration
- **Log-scale search**: Use small ranges like `[1e-5,1e-2]` for learning rates

### 3. Common Range Recommendations

**Conservative (Faster optimization):**
- BATCH_SIZE: `[64,256]`
- MAX_EPOCHS: `[20,100]`
- HIDDEN_UNITS: `[16,64]`

**Aggressive (More thorough):**
- BATCH_SIZE: `[32,512]` 
- MAX_EPOCHS: `[50,300]`
- HIDDEN_UNITS: `[8,256]`

## Validation Rules

The system will reject configurations with:
- ❌ Comma-separated lists in Optuna mode: `"BATCH_SIZE": "100,200,400"`
- ❌ Missing required parameters for your model type
- ❌ Invalid bracket format: `"LAYERS": "[1-3]"` (should be `"[1,3]"`)
- ❌ Multiple optimizers: `"OPTIMIZER_TYPE": "Adam,SGD"`

## Tips for Effective Optimization

1. **Start Conservative**: Use smaller ranges for initial experiments
2. **Use Log Scale**: For learning rates, always use wide ranges like `[1e-5,1e-2]`
3. **Model Architecture**: Don't make ranges too wide for layers/units initially
4. **Validation Patience**: Usually keep fixed around 10-20 epochs
5. **Batch Size**: Powers of 2 work best: `[32,64,128,256,512]`

## Example Complete Configuration

```json
{
    "FEATURE_COLUMNS": ["Power", "Battery_Temp_degC", "SOC"],
    "TARGET_COLUMN": "Voltage", 
    "MODEL_TYPE": "GRU",
    "TRAINING_METHOD": "Sequence-to-Sequence",
    "DEVICE_SELECTION": "cuda:0",
    "OPTIMIZER_TYPE": "Adam",
    "SCHEDULER_TYPE": "CosineAnnealingWarmRestarts",
    
    "BATCH_SIZE": "[128,512]",
    "MAX_EPOCHS": "[50,200]",
    "INITIAL_LR": "[1e-5,1e-2]", 
    "GRU_LAYERS": "[1,3]",
    "GRU_HIDDEN_UNITS": "[32,128]",
    "LOOKBACK": "[200,600]",
    "COSINE_T0": "[10,30]",
    "COSINE_T_MULT": "[1,3]",
    
    "VALID_PATIENCE": "15",
    "REPETITIONS": 1
}
```
