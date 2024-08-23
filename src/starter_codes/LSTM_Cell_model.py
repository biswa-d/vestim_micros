#----------------------------------------------------------------------------------------
#Description: This script contains the code for the backend generation of LSTM models using the hyper parameters entered by the user
# to the GUI interface.
#
# Created on Wed Aug 07 2024 15:57:04
# Author: Biswanath Dehury
# Company: Dr. Phil Kollmeyer's Battery Lab at McMaster University
# Copyright (c) 2024 Biswanath Dehury, Dr. Phil Kollmeyer's Battery Lab at McMaster University
# testing line to check git
#----------------------------------------------------------------------------------------

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_units, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_units, 1)  # Assuming regression, adjust for your use case

    def forward(self, x, h_s, h_c):
        # If h_s and h_c are not provided, initialize them
        if h_s is None or h_c is None:
            h_s = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
            h_c = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        
        out, (h_s, h_c) = self.lstm(x, (h_s, h_c))
        out = self.fc(out[:, -1, :])
        return out, (h_s, h_c)

def build_lstm_model(params):
    input_size = params.get("INPUT_SIZE", 3)  # Default input size set to 3, change if needed
    hidden_units = params["HIDDEN_UNITS"]  # Updated from HIDDEN_SIZE to HIDDEN_UNITS
    num_layers = params["LAYERS"]
    return LSTMModel(input_size, hidden_units, num_layers)


