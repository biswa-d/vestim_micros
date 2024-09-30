import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

class HLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_units, hidden_gate_units):
        super(HLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.hidden_gate_units = hidden_gate_units

        # Input gate layers (with extra hidden layer)
        self.input_gate_fc1 = nn.Linear(input_size + hidden_units, hidden_gate_units)
        self.input_gate_fc2 = nn.Linear(hidden_gate_units, hidden_units)
        
        # Forget gate layers
        self.forget_gate_fc1 = nn.Linear(input_size + hidden_units, hidden_gate_units)
        self.forget_gate_fc2 = nn.Linear(hidden_gate_units, hidden_units)

        # Output gate layers
        self.output_gate_fc1 = nn.Linear(input_size + hidden_units, hidden_gate_units)
        self.output_gate_fc2 = nn.Linear(hidden_gate_units, hidden_units)

        # Cell state layers
        self.cell_state_fc1 = nn.Linear(input_size + hidden_units, hidden_gate_units)
        self.cell_state_fc2 = nn.Linear(hidden_gate_units, hidden_units)

    def forward(self, x, h, c):
        # Concatenate input and hidden state
        combined = torch.cat([x, h], dim=1)

        # Compute input, forget, and output gates with hidden layers
        input_gate = torch.sigmoid(self.input_gate_fc2(torch.relu(self.input_gate_fc1(combined))))
        forget_gate = torch.sigmoid(self.forget_gate_fc2(torch.relu(self.forget_gate_fc1(combined))))
        output_gate = torch.sigmoid(self.output_gate_fc2(torch.relu(self.output_gate_fc1(combined))))

        # Compute the cell state update
        cell_state = torch.tanh(self.cell_state_fc2(torch.relu(self.cell_state_fc1(combined))))
        c_next = forget_gate * c.clone() + input_gate * cell_state

        # Compute the hidden state output
        h_next = output_gate * torch.tanh(c_next)

        return h_next, c_next

class HLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, hidden_gate_units, device):
        super(HLSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_gate_units = hidden_gate_units
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device

        # Stack multiple HLSTM cells
        self.cells = nn.ModuleList([HLSTMCell(input_size, hidden_units, hidden_gate_units).to(device)
                                    for _ in range(num_layers)])

        # Output layer
        self.fc = nn.Linear(hidden_units, 1).to(device)

    def forward(self, x, h_s=None, c_s=None):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Initialize hidden states if not provided
        if h_s is None:
            h_s = [torch.zeros(batch_size, self.hidden_units).to(self.device) for _ in range(self.num_layers)]
        if c_s is None:
            c_s = [torch.zeros(batch_size, self.hidden_units).to(self.device) for _ in range(self.num_layers)]

        # Process the input sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            for layer in range(self.num_layers):
                # Create new tensors for h_next and c_next
                h_next, c_next = self.cells[layer](x_t, h_s[layer], c_s[layer])
                h_s[layer] = h_next  # Assign new tensor to avoid in-place operation
                c_s[layer] = c_next
                x_t = h_s[layer]

        # Use the hidden state of the last layer as output
        out = self.fc(h_s[-1])
        return out, h_s, c_s
    
    def apply_pruning(self, drop_out=0.2):
        """Apply pruning to the layers"""
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.LSTM):
                prune.l1_unstructured(layer, name='weight', amount=drop_out)  # Prune 20% of the connections

    def remove_pruning(self):
        """Remove pruning from the layers"""
        for name, layer in self.named_modules():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.LSTM):
                prune.remove(layer, 'weight')


class HLSTMModelService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_hlstm_model(self, params):
        """
        Build the HLSTM model using the provided parameters.

        :param params: Dictionary containing model parameters.
        :return: An instance of HLSTMModel.
        """
        input_size = params.get("INPUT_SIZE", 3)  # Default input size set to 3, change if needed
        hidden_units = int(params["HIDDEN_UNITS"])  # Ensure hidden_units is an integer
        num_layers = int(params["LAYERS"])
        hidden_gate_units = int(params.get("HIDDEN_GATE_UNITS", 32))  # Add hidden gate units

        print(f"Building HLSTM model with input_size={input_size}, hidden_units={hidden_units}, "
              f"num_layers={num_layers}, hidden_gate_units={hidden_gate_units}")

        # Create an instance of HLSTMModel
        model = HLSTMModel(
            input_size=input_size,
            hidden_units=hidden_units,
            num_layers=num_layers,
            hidden_gate_units=hidden_gate_units,
            device=self.device
        )

        return model

    def save_model(self, model, model_path):
        """
        Save the model to the specified path after removing pruning reparameterizations.

        :param model: The PyTorch model to save.
        :param model_path: The file path where the model will be saved.
        """
        # Remove pruning reparameterizations before saving
        # model.remove_pruning()
        # Save the model's state dictionary
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    def create_and_save_hlstm_model(self, params, model_path):
        """
        Build and save an HLSTM model using the provided parameters.

        :param params: Dictionary containing model parameters.
        :param model_path: The file path where the model will be saved.
        :return: The built HLSTM model.
        """
        model = self.build_hlstm_model(params)
        self.save_model(model, model_path)
        return model
    

