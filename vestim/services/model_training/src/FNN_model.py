import torch
import torch.nn as nn

class FNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes, dropout_prob=0.0, apply_clipped_relu=False):
        """
        A simple Feedforward Neural Network (FNN/MLP).

        :param input_size: Number of input features.
        :param output_size: Number of output features (typically 1 for regression).
        :param hidden_layer_sizes: A list of integers, where each integer is the
                                   number of neurons in a hidden layer.
                                   Example: [128, 64, 32] for three hidden layers.
        :param dropout_prob: Dropout probability to apply after each hidden layer.
        :param apply_clipped_relu: If True, applies a ReLU clipped at 1.0 to the output.
        """
        super(FNNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_prob = dropout_prob
        self.apply_clipped_relu = apply_clipped_relu

        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            current_input_size = hidden_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        if self.apply_clipped_relu:
            layers.append(nn.ReLU(inplace=True)) # Using a standard ReLU and then clipping, or a custom lambda
            # This is a simple way to implement it. A custom lambda could also be used.
            # Forcing output to be between 0 and 1.
            layers.append(torch.nn.Hardtanh(min_val=0, max_val=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the FNN.
        :param x: Input tensor of shape [batch_size, input_size]
        :return: Output tensor of shape [batch_size, output_size]
        """
        return self.network(x)