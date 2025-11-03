import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class LSTMStacked(nn.Module):
    """
    Stacked LSTM with per-layer hidden sizes, e.g., [64, 32].
    Builds a ModuleList of single-layer LSTMs where each layer can have
    a different hidden size. Supports optional carry of hidden states as
    lists of tensors (one per layer), or None for zero init.
    """
    def __init__(self, input_size: int, layer_sizes: List[int], device: str = 'cpu',
                 dropout_prob: float = 0.0, apply_clipped_relu: bool = False):
        super().__init__()
        self.input_size = input_size
        self.layer_sizes = [int(x) for x in layer_sizes]
        self.num_layers = len(self.layer_sizes)
        self.hidden_units = self.layer_sizes[-1]
        self.device = torch.device(device)
        self.dropout_prob = float(dropout_prob)
        self.apply_clipped_relu = apply_clipped_relu

        lstms = []
        in_size = input_size
        for h in self.layer_sizes:
            lstm = nn.LSTM(input_size=in_size, hidden_size=h, num_layers=1, batch_first=True)
            lstms.append(lstm)
            in_size = h
        self.layers = nn.ModuleList(lstms).to(self.device)

        self.dropout = nn.Dropout(p=self.dropout_prob) if self.dropout_prob > 0 else nn.Identity()
        self.fc = nn.Linear(self.hidden_units, 1).to(self.device)
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

    def _zero_hidden(self, batch_size: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        h_list, c_list = [], []
        for h in self.layer_sizes:
            h_list.append(torch.zeros(1, batch_size, h, device=self.device))
            c_list.append(torch.zeros(1, batch_size, h, device=self.device))
        return h_list, c_list

    @staticmethod
    def _detach_hidden_list(h_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [t.detach() for t in h_list]

    def forward(self, x: torch.Tensor, h_s=None, h_c=None):
        x = x.to(self.device)
        batch = x.size(0)

        # Normalize hidden inputs: allow None or lists
        if h_s is None or h_c is None:
            h_s_list, h_c_list = self._zero_hidden(batch)
        else:
            # Accept either tensors (shape [num_layers, B, H]) if sizes uniform, or lists
            if isinstance(h_s, torch.Tensor) and isinstance(h_c, torch.Tensor):
                # Split tensor across layers (assume same hidden size) - keep only matching dim
                h_s_list = [h_s[i:i+1].to(self.device) for i in range(self.num_layers)]
                h_c_list = [h_c[i:i+1].to(self.device) for i in range(self.num_layers)]
            elif isinstance(h_s, list) and isinstance(h_c, list):
                h_s_list = [t.to(self.device) for t in h_s]
                h_c_list = [t.to(self.device) for t in h_c]
            else:
                raise TypeError("Unsupported hidden state format for LSTMStacked")

        out = x
        new_h_s, new_h_c = [], []
        for i, lstm in enumerate(self.layers):
            out, (h_s_i, h_c_i) = lstm(out, (h_s_list[i], h_c_list[i]))
            # Dropout between layers (except maybe last)
            if i < self.num_layers - 1:
                out = self.dropout(out)
            new_h_s.append(h_s_i)
            new_h_c.append(h_c_i)

        out = self.fc(out[:, -1, :])
        out = self.final_activation(out)
        return out, (new_h_s, new_h_c)

    def detach_hidden(self, hidden):
        h_s, h_c = hidden
        if isinstance(h_s, list):
            return ([t.detach() for t in h_s], [t.detach() for t in h_c])
        else:
            return (h_s.detach(), h_c.detach())
