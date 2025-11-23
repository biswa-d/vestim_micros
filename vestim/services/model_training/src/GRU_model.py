import torch
import torch.nn as nn

class GRULayerNorm(nn.Module):
    """A custom GRU layer with Layer Normalization."""
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRULayerNorm, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, hx):
        x, hx = self.gru(x, hx)
        x = self.layer_norm(x)
        return x, hx

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, output_size=1, dropout_prob=0.0, device='cpu', apply_clipped_relu=False, use_layer_norm=False):
        """
        Gated Recurrent Unit (GRU) Model.

        :param input_size: Number of input features.
        :param hidden_units: Number of features in the hidden state h.
        :param num_layers: Number of recurrent layers.
        :param output_size: Number of output features (typically 1 for regression).
        :param dropout_prob: Dropout probability for GRU layers (if num_layers > 1) and an optional final dropout.
        :param device: The device to run the model on ('cpu' or 'cuda').
        :param apply_clipped_relu: If True, applies a ReLU clipped at 1.0 to the output.
        :param use_layer_norm: If True, adds Layer Normalization to the GRU layer.
        """
        super(GRUModel, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_prob = dropout_prob
        self.device = device
        self.apply_clipped_relu = apply_clipped_relu
        self.use_layer_norm = use_layer_norm

        if self.use_layer_norm:
            self.gru = GRULayerNorm(
                input_size=input_size,
                hidden_size=hidden_units,
                num_layers=num_layers,
                dropout=dropout_prob if num_layers > 1 else 0.0
            ).to(self.device)
        else:
            self.gru = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_units,
                num_layers=num_layers,
                batch_first=True, # Expects input: (batch, seq, feature)
                dropout=dropout_prob if num_layers > 1 else 0.0  # Dropout only between GRU layers
            ).to(self.device)

        # NO additional output dropout layer - dropout is already handled between GRU layers
        self.fc = nn.Linear(hidden_units, output_size).to(self.device)
        
        if self.apply_clipped_relu:
            self.final_activation = torch.nn.Hardtanh(min_val=0, max_val=1)
        else:
            self.final_activation = nn.Identity()
        
        # Initialize weights properly to prevent gradient issues
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot initialization to prevent gradient issues."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Bias terms
                nn.init.constant_(param.data, 0.0)
        
        # Initialize the fully connected layer
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def forward(self, x, h_0=None):
        """
        Forward pass for the GRU model.

        :param x: Input tensor of shape [batch_size, sequence_length, input_size].
        :param h_0: Initial hidden state of shape [num_layers, batch_size, hidden_units].
                    If None, it will be initialized to zeros.
        :return: Output tensor of shape [batch_size, sequence_length, output_size] (if fc applied to all outputs)
                 or [batch_size, output_size] (if fc applied to last output only),
                 and the final hidden state.
                 For simplicity and common use in sequence-to-one or sequence-to-sequence where
                 only the last output's prediction or all outputs are processed by fc,
                 this example applies fc to all outputs of GRU.
        """
        x = x.to(self.device)
        
        # Check for NaN or infinite values in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError("Input contains NaN or infinite values")
        
        if h_0 is None:
            # Initialize hidden state if not provided
            h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_units, device=self.device)
        else:
            h_0 = h_0.to(self.device)
            # Check for NaN or infinite values in hidden state
            if torch.isnan(h_0).any() or torch.isinf(h_0).any():
                raise ValueError("Hidden state contains NaN or infinite values")

        # GRU output: output features for each time step, and the final hidden state
        out, h_n = self.gru(x, h_0)
        # out shape: (batch_size, seq_len, hidden_units)
        # h_n shape: (num_layers, batch_size, hidden_units)

        # Apply the fully connected layer to the last time step only (for sequence-to-one prediction)
        # This matches the behavior of LSTMModel and avoids shape mismatches during training
        out = self.fc(out[:, -1, :])  # Shape: (batch_size, output_size)
        
        out = self.final_activation(out)
        
        # Check for NaN or infinite values in output
        if torch.isnan(out).any() or torch.isinf(out).any():
            raise ValueError("Model output contains NaN or infinite values")

        return out, h_n