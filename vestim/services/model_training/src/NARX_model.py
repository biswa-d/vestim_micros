import torch
import torch.nn as nn


class NARXModel(nn.Module):
    """
    Nonlinear Autoregressive model with eXogenous input (NARX) using PyTorch.
    
    NARX combines both historical outputs (autoregressive) and exogenous inputs
    to make predictions: y(t) = f(y(t-1), ..., y(t-p), x(t), ..., x(t-s))
    
    This implementation uses feedforward layers to combine:
    - Previous output value(s) (autoregressive component)
    - Current and historical exogenous features
    """
    
    def __init__(self, input_size, output_size, hidden_layer_sizes, output_delay=1, 
                 dropout_prob=0.0, apply_clipped_relu=False, activation_function='ReLU', 
                 use_layer_norm=False, device='cpu'):
        """
        Initialize NARX model.
        
        :param input_size: Number of exogenous input features at current timestep.
        :param output_size: Number of output features (typically 1 for regression).
        :param hidden_layer_sizes: List of hidden layer sizes.
        :param output_delay: Number of previous output timesteps to use (AR order).
        :param dropout_prob: Dropout probability.
        :param apply_clipped_relu: Whether to clip output to [0,1] range.
        :param activation_function: Activation function name ('ReLU' or 'GELU').
        :param use_layer_norm: Whether to use layer normalization.
        :param device: Device to place model on ('cpu' or 'cuda').
        """
        super(NARXModel, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.output_delay = output_delay  # How many previous outputs to use (AR order)
        self.dropout_prob = dropout_prob
        self.apply_clipped_relu = apply_clipped_relu
        self.activation_function = activation_function
        self.use_layer_norm = use_layer_norm
        self.device = device
        
        # Total input to the network: exogenous inputs + previous outputs
        # At each timestep: [x(t), y(t-1), y(t-2), ..., y(t-output_delay)]
        narx_input_size = input_size + (output_delay * output_size)
        
        layers = []
        current_input_size = narx_input_size
        
        activation_fn = nn.GELU() if self.activation_function == 'GELU' else nn.ReLU()
        
        # Build hidden layers
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation_fn)
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            current_input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_input_size, output_size))
        
        # Optional clipped ReLU output
        if self.apply_clipped_relu:
            layers.append(torch.nn.Hardtanh(min_val=0, max_val=1))
        
        self.network = nn.Sequential(*layers)
        
        # Xavier initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using Xavier/Glorot uniform initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, x_current, y_previous=None):
        """
        Forward pass for NARX model.
        
        :param x_current: Current exogenous inputs, shape [batch_size, input_size]
        :param y_previous: Previous output values, shape [batch_size, output_delay * output_size]
                          If None, initialized to zeros.
        :return: Predicted output, shape [batch_size, output_size]
        """
        x_current = x_current.to(self.device)
        
        # Initialize previous outputs if not provided
        if y_previous is None:
            y_previous = torch.zeros(
                x_current.size(0), 
                self.output_delay * self.output_size,
                device=self.device
            )
        else:
            y_previous = y_previous.to(self.device)
        
        # Concatenate exogenous input with previous outputs
        # NARX input = [x(t), y(t-1), y(t-2), ..., y(t-output_delay)]
        narx_input = torch.cat([x_current, y_previous], dim=1)
        
        # Forward through network
        output = self.network(narx_input)
        
        return output
    
    def forward_sequence(self, x_sequence, y_init=None):
        """
        Forward pass for a sequence of exogenous inputs, generating one output per timestep.
        This is useful for sequential prediction where we accumulate previous outputs.
        
        :param x_sequence: Sequence of exogenous inputs, shape [batch_size, seq_len, input_size]
        :param y_init: Initial output values for warmup, shape [batch_size, output_delay * output_size]
        :return: Predicted sequence, shape [batch_size, seq_len, output_size]
        """
        batch_size = x_sequence.size(0)
        seq_len = x_sequence.size(1)
        
        # Initialize previous outputs
        if y_init is None:
            y_previous = torch.zeros(
                batch_size, 
                self.output_delay * self.output_size,
                device=self.device
            )
        else:
            y_previous = y_init.to(self.device)
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x_sequence[:, t, :]  # [batch_size, input_size]
            
            # Get prediction for this timestep
            y_t = self.forward(x_t, y_previous)  # [batch_size, output_size]
            outputs.append(y_t)
            
            # Update y_previous: shift and append new prediction
            # Remove oldest output, add newest prediction
            if self.output_delay > 1:
                y_previous = torch.cat([
                    y_t,  # New prediction
                    y_previous[:, :-self.output_size]  # Remove oldest
                ], dim=1)
            else:
                y_previous = y_t
        
        # Stack all outputs along time dimension
        output_sequence = torch.stack(outputs, dim=1)  # [batch_size, seq_len, output_size]
        
        return output_sequence
