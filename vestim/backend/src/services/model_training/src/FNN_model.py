import torch
import torch.nn as nn

class FNNModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layer_sizes, dropout_prob=0.0):
        """
        A simple Feedforward Neural Network (FNN/MLP).

        :param input_size: Number of input features.
        :param output_size: Number of output features (typically 1 for regression).
        :param hidden_layer_sizes: A list of integers, where each integer is the
                                   number of neurons in a hidden layer.
                                   Example: [128, 64, 32] for three hidden layers.
        :param dropout_prob: Dropout probability to apply after each hidden layer.
        """
        super(FNNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_prob = dropout_prob

        layers = []
        current_input_size = input_size
        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(current_input_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
            current_input_size = hidden_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass for the FNN.
        :param x: Input tensor of shape [batch_size, input_size]
        :return: Output tensor of shape [batch_size, output_size]
        """
        return self.network(x)