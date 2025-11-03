import torch
import torch.nn as nn
from typing import List, Optional


class GRUStacked(nn.Module):
    """
    Stacked GRU with per-layer hidden sizes, e.g., [64, 32].
    Uses a ModuleList of one-layer GRUs to allow variable sizes.
    Hidden state is a list of tensors (one per layer) when sizes differ.
    """
    def __init__(self, input_size: int, layer_sizes: List[int], output_size: int = 1,
                 dropout_prob: float = 0.0, device: str = 'cpu', apply_clipped_relu: bool = False):
        super().__init__()
        self.input_size = input_size
        self.layer_sizes = [int(x) for x in layer_sizes]
        self.num_layers = len(self.layer_sizes)
        self.hidden_units = self.layer_sizes[-1]
        self.output_size = output_size
        self.device = torch.device(device)
        self.dropout_prob = float(dropout_prob)
        self.apply_clipped_relu = apply_clipped_relu

        grus = []
        in_size = input_size
        for h in self.layer_sizes:
            grus.append(nn.GRU(input_size=in_size, hidden_size=h, num_layers=1, batch_first=True))
            in_size = h
        self.layers = nn.ModuleList(grus).to(self.device)

        self.dropout = nn.Dropout(p=self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()
        self.fc = nn.Linear(self.hidden_units, self.output_size).to(self.device)
        self.final_activation = torch.nn.Hardtanh(min_val=0, max_val=1) if self.apply_clipped_relu else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        for m in self.layers:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0.0)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def _zero_hidden(self, batch_size: int):
        return [torch.zeros(1, batch_size, h, device=self.device) for h in self.layer_sizes]

    def forward(self, x, h_0=None):
        x = x.to(self.device)
        batch = x.size(0)
        if h_0 is None:
            h_list = self._zero_hidden(batch)
        else:
            if isinstance(h_0, torch.Tensor):
                h_list = [h_0[i:i+1].to(self.device) for i in range(self.num_layers)]
            elif isinstance(h_0, list):
                h_list = [t.to(self.device) for t in h_0]
            else:
                raise TypeError("Unsupported hidden state format for GRUStacked")

        out = x
        new_h = []
        for i, gru in enumerate(self.layers):
            out, h_i = gru(out, h_list[i])
            if i < self.num_layers - 1:
                out = self.dropout(out)
            new_h.append(h_i)

        out = self.fc(out[:, -1, :])
        out = self.final_activation(out)
        return out, new_h

    @staticmethod
    def detach_hidden(hidden):
        if isinstance(hidden, list):
            return [t.detach() for t in hidden]
        return hidden.detach()
