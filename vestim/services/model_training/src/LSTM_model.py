import torch
import torch.nn as nn

class LSTMLayerNorm(nn.Module):
    """A custom LSTM layer with Layer Normalization."""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMLayerNorm, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hx):
        x, hx = self.lstm(x, hx)
        x = self.layer_norm(x)
        return x, hx

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device, dropout_prob=0.0, apply_clipped_relu=False, use_layer_norm=False):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.dropout_prob = dropout_prob
        self.apply_clipped_relu = apply_clipped_relu
        self.use_layer_norm = use_layer_norm

        # Define the LSTM layer with dropout between layers
        if self.use_layer_norm:
            self.lstm = LSTMLayerNorm(
                input_size,
                hidden_units,
                num_layers,
                dropout=dropout_prob if num_layers > 1 else 0
            ).to(self.device)
        else:
            self.lstm = nn.LSTM(
                input_size,
                hidden_units,
                num_layers,
                batch_first=True,
                dropout=dropout_prob if num_layers > 1 else 0
            ).to(self.device)

        # Define a dropout layer for the outputs
        self.dropout = nn.Dropout(p=dropout_prob)

        # Define the fully connected layer
        self.fc = nn.Linear(hidden_units, 1).to(self.device)
        
        # Define the final activation layer
        if self.apply_clipped_relu:
            self.final_activation = torch.nn.Hardtanh(min_val=0, max_val=1)
        else:
            self.final_activation = nn.Identity()


    def forward(self, x, h_s=None, h_c=None):
        # Ensure the input is on the correct device
        x = x.to(self.device)

        # Pass input through LSTM
        out, (h_s, h_c) = self.lstm(x, (h_s, h_c))

        # Apply dropout to the outputs of the LSTM
        out = self.dropout(out)

        # Pass the output through the fully connected layer
        # Apply the fully connected layer to the last time step only (for sequence-to-one prediction)
        out = self.fc(out[:, -1, :])
        
        out = self.final_activation(out)

        return out, (h_s, h_c)