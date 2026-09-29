# Model Loading Instructions

## Model Details
- Model Type: LSTM
- Input Size: 3
- Hidden Units: 18
- Layers: 2
- Output Size: 1
- Lookback: 400

## Feature Configuration
- Input Features: Power, Battery_Temp_degC, SOC
- Target Variable: Voltage

## Loading Options

### Option 1: Using VEstim Environment
```python
import torch
from vestim.services.model_training.src.LSTM_model_service_test import LSTMModelService

# Load the exported model
checkpoint = torch.load('model_export.pt')

# Create model instance based on model type
model_type = checkpoint['model_type']
if model_type in ['LSTM', 'GRU']:
    model_service = LSTMModelService()
    model = model_service.create_model(
        input_size=checkpoint['hyperparams']['input_size'],
        hidden_size=checkpoint['hyperparams']['hidden_size'],
        num_layers=checkpoint['hyperparams']['num_layers'],
        output_size=checkpoint['hyperparams']['output_size']
    )
elif model_type == 'FNN':
    from vestim.services.model_training.src.FNN_model_service import FNNModelService
    model_service = FNNModelService()
    model = model_service.create_model(
        input_size=checkpoint['hyperparams']['input_size'],
        hidden_layer_sizes=checkpoint['hyperparams']['hidden_layer_sizes'],
        output_size=checkpoint['hyperparams']['output_size'],
        dropout_prob=checkpoint['hyperparams']['dropout_prob']
    )

# Load state dict
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # Set to evaluation mode
```

### Option 2: Standalone Usage (No VEstim Required)
```python
import torch
import torch.nn as nn

# Load the checkpoint
checkpoint = torch.load('model_export.pt')

# Execute the model definition code (included in the checkpoint)
exec(checkpoint['model_definition'])

# Create model instance based on model type
model_type = checkpoint['model_type']
if model_type in ['LSTM', 'GRU']:
    model = LSTMModel(
        input_size=checkpoint['hyperparams']['input_size'],
        hidden_units=checkpoint['hyperparams']['hidden_size'],
        num_layers=checkpoint['hyperparams']['num_layers'],
        output_size=checkpoint['hyperparams']['output_size']
    )
elif model_type == 'FNN':
    # For FNN, you would need to execute the FNN model definition
    # and create an FNN model instance with appropriate parameters
    model = FNNModel(
        input_size=checkpoint['hyperparams']['input_size'],
        hidden_layer_sizes=checkpoint['hyperparams']['hidden_layer_sizes'],
        output_size=checkpoint['hyperparams']['output_size'],
        dropout_prob=checkpoint['hyperparams']['dropout_prob']
    )

# Load state dict
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Example usage:
def predict(model, input_data):
    with torch.no_grad():
        if model_type in ['LSTM', 'GRU']:
            output, _ = model(input_data)
        else:  # FNN
            output = model(input_data)
    return output
```

## Input Data Format
- Input shape should be: (batch_size, lookback, input_size)
- Features should be in order: Power, Battery_Temp_degC, SOC
- All inputs should be normalized using the same scaling as training data

## Example Preprocessing
```python
import numpy as np

def preprocess_data(data, lookback=400):
    # Ensure data is normalized using the same scaling as training
    # Create sequences of length 'lookback'
    sequences = []
    for i in range(len(data) - lookback + 1):
        sequences.append(data[i:(i + lookback)])
    return torch.FloatTensor(np.array(sequences))
```

## Making Predictions
```python
# Example prediction
input_sequence = preprocess_data(your_data)  # Shape: (1, lookback, input_size)
with torch.no_grad():
    prediction, _ = model(input_sequence)
```
