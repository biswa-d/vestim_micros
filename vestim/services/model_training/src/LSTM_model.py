import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device, dropout_prob=0.0, apply_clipped_relu=False, use_layer_norm=False):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.dropout_prob = dropout_prob
        self.apply_clipped_relu = apply_clipped_relu
        # Define the LSTM layer with dropout between layers (no layer norm, reference code)
        self.lstm = nn.LSTM(
            input_size,
            hidden_units,
            num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0  # Dropout only between LSTM layers
        ).to(self.device)

        # Define the fully connected layer (NO additional output dropout - already handled by LSTM)
        self.fc = nn.Linear(hidden_units, 1).to(self.device)
        
        # No output activation (reference code: Junran Chen)
        
        # Initialize weights properly to prevent gradient issues and flat loss
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights using Xavier/Glorot initialization to prevent gradient issues.
        PyTorch's default Kaiming uniform can cause dead neurons with normalized data,
        leading to probabilistic flat loss behavior (outputs stuck near dataset mean).
        """
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal for RNNs (prevents vanishing/exploding gradients)
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # LSTM bias initialization: set forget gate bias to 1.0 for better gradient flow
                # LSTM has 4 gates (input, forget, output, cell), biases are concatenated
                # bias_ih and bias_hh both have shape [4*hidden_size]
                hidden_size = param.data.size(0) // 4
                # Initialize all biases to 0
                nn.init.constant_(param.data, 0.0)
                # Set forget gate bias to 1.0 (indices hidden_size to 2*hidden_size)
                param.data[hidden_size:2*hidden_size].fill_(1.0)
        
        # Initialize the fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.01)  # Small positive bias for better gradient flow

    def forward(self, x, h_s=None, h_c=None):
        # Ensure the input is on the correct device
        x = x.to(self.device)

        # Pass input through LSTM
        # Allow None hidden states: let PyTorch initialize zeros for us
        if h_s is None or h_c is None:
            out, (h_s, h_c) = self.lstm(x)
        else:
            out, (h_s, h_c) = self.lstm(x, (h_s, h_c))

        # Pass the output through the fully connected layer
        # Apply the fully connected layer to the last time step only (for sequence-to-one prediction)
        out = self.fc(out[:, -1, :])
        # No clamp, ReLU, or LeakyReLU on output (reference code)
        return out, (h_s, h_c)