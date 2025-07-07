# FNN Model Regularization Summary

## Overview
This document summarizes the regularization techniques implemented in the FNN (Feedforward Neural Network) model to prevent overfitting and improve generalization.

## Implemented Regularization Techniques

### 1. Dropout Regularization
- **File**: `vestim/services/model_training/src/FNN_model.py`
- **Implementation**: Applied after each hidden layer's ReLU activation
- **Configuration**: Controlled by `DROPOUT_PROB` hyperparameter (default: 0.0)
- **How it works**: Randomly sets a fraction of input units to 0 during training, preventing co-adaptation of neurons
- **Code location**: Lines 26-27 in FNN_model.py

```python
if dropout_prob > 0:
    layers.append(nn.Dropout(dropout_prob))
```

### 2. Weight Decay (L2 Regularization)
- **File**: `vestim/gateway/src/training_task_manager_qt.py`
- **Implementation**: Added to Adam optimizer as weight_decay parameter
- **Configuration**: Controlled by `WEIGHT_DECAY` hyperparameter (default: 0.0)
- **How it works**: Adds L2 penalty to loss function, encouraging smaller weights
- **Code location**: Line 402 in training_task_manager_qt.py

```python
weight_decay = hyperparams.get('WEIGHT_DECAY', 0.0)
self.optimizer = torch.optim.Adam(model.parameters(), lr=current_lr, weight_decay=weight_decay)
```

## Hyperparameter Configuration

### Updated hyperparams.json
The default hyperparameters now include regularization options:

```json
{
    "WEIGHT_DECAY": "0.0",     // L2 regularization strength
    "DROPOUT_PROB": "0.0"      // Dropout probability (0.0 to 1.0)
}
```

### Recommended Values
- **DROPOUT_PROB**: 
  - Start with 0.1-0.3 for moderate regularization
  - Use 0.5 for strong regularization
  - Set to 0.0 to disable dropout
  
- **WEIGHT_DECAY**:
  - Start with 1e-4 to 1e-5 for moderate L2 regularization
  - Use 1e-3 for stronger regularization
  - Set to 0.0 to disable weight decay

## Usage Examples

### Light Regularization (recommended starting point)
```json
{
    "DROPOUT_PROB": "0.1",
    "WEIGHT_DECAY": "1e-4"
}
```

### Moderate Regularization
```json
{
    "DROPOUT_PROB": "0.3",
    "WEIGHT_DECAY": "1e-3"
}
```

### Strong Regularization (for overfitting models)
```json
{
    "DROPOUT_PROB": "0.5",
    "WEIGHT_DECAY": "1e-2"
}
```

## Benefits

1. **Dropout**:
   - Reduces overfitting by preventing neuron co-adaptation
   - Improves model generalization
   - Acts as ensemble method during training

2. **Weight Decay**:
   - Prevents weights from growing too large
   - Improves model stability
   - Reduces overfitting through L2 penalty

## Implementation Notes

- Dropout is automatically disabled during evaluation (handled by PyTorch)
- Weight decay is applied during training through the optimizer
- Both techniques can be used together for enhanced regularization
- Start with conservative values and increase if overfitting persists

## Testing Regularization

To test the effectiveness of regularization:

1. Train models with and without regularization
2. Compare training vs validation loss curves
3. Look for reduced gap between training and validation performance
4. Monitor for improved generalization on test data

## Future Enhancements

Potential additional regularization techniques to consider:

1. **Batch Normalization**: Normalize inputs to each layer
2. **Layer Normalization**: Alternative to batch normalization
3. **Early Stopping**: Already implemented via VALID_PATIENCE
4. **Data Augmentation**: Already implemented in data handlers
5. **Learning Rate Scheduling**: Already implemented via LR_DROP_*
