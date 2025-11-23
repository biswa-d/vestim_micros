import torch
import torch.nn as nn
import torch.nn.functional as F

class EmaHead(nn.Module):
    def __init__(self, init_alpha=0.1, adaptive=False, feat_dim=0):
        super().__init__()
        self.adaptive = adaptive
        if adaptive:
            self.alpha_mlp = nn.Sequential(
                nn.Linear(feat_dim, 16), nn.ReLU(), nn.Linear(16, 1)
            )
        else:
            # keep alpha in (0,1) via sigmoid of a free parameter
            self.logit_alpha = nn.Parameter(torch.logit(torch.tensor(init_alpha)))
    def forward(self, r, feat=None):   # r: [T,B,1]; feat (if adaptive): [T,B,feat_dim]
        T,B,_ = r.shape
        y = torch.zeros_like(r)
        if self.adaptive:
            # optional clamp to avoid extreme alphas
            alpha = torch.sigmoid(self.alpha_mlp(feat)).clamp(0.01, 0.99)  # [T,B,1]
        else:
            alpha = torch.sigmoid(self.logit_alpha).view(1,1,1).expand_as(r)
        y[0] = r[0]
        for t in range(1, T):                      # differentiable scan
            y[t] = (1 - alpha[t]) * y[t-1] + alpha[t] * r[t]
        return y

class CausalLPFHead(nn.Module):
    def __init__(self, K=15):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(K))

    def forward(self, r, z=None):  # r: [T,B,1], z: [B,K-1,1]
        w = torch.softmax(self.logits, dim=0).view(1, 1, -1)
        x = r.transpose(0, 1)
        if z is None:
            z = torch.zeros(x.shape[0], w.shape[-1] - 1, x.shape[-1], device=x.device)
        x = torch.cat([z, x], dim=1)
        y = F.conv1d(x.transpose(1, 2), w).transpose(1, 2)
        return y.transpose(0, 1), x[:, -w.shape[-1] + 1:]

class LSTM_EMA(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device, dropout_prob=0.0, apply_clipped_relu=False):
        super(LSTM_EMA, self).__init__()
        self.lstm = LSTMModel(input_size, hidden_units, num_layers, device, dropout_prob, apply_clipped_relu)
        self.ema_head = EmaHead()

    @property
    def hidden_units(self):
        return self.lstm.hidden_units

    @property
    def num_layers(self):
        return self.lstm.num_layers

    def forward(self, x, h_s=None, h_c=None):
        out, (h_s, h_c) = self.lstm(x, h_s, h_c)
        out = out.unsqueeze(0) # Reshape for EmaHead
        out = self.ema_head(out)
        out = out.squeeze(0)
        return out, (h_s, h_c)

class LSTM_LPF(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device, dropout_prob=0.0, apply_clipped_relu=False):
        super(LSTM_LPF, self).__init__()
        self.lstm = LSTMModel(input_size, hidden_units, num_layers, device, dropout_prob, apply_clipped_relu)

    @property
    def hidden_units(self):
        return self.lstm.hidden_units

    @property
    def num_layers(self):
        return self.lstm.num_layers

    def forward(self, x, h_s=None, h_c=None, z=None):
        # The 'z' parameter is kept for compatibility with the testing service, but it is not used.
        out, (h_s, h_c) = self.lstm(x, h_s, h_c)
        return out, (h_s, h_c), z

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_units, num_layers, device, dropout_prob=0.0, apply_clipped_relu=False):
        super(LSTMModel, self).__init__()
        self.input_size = input_size
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.device = device
        self.dropout_prob = dropout_prob
        self.apply_clipped_relu = apply_clipped_relu

        self.lstm = nn.LSTM(
            input_size,
            hidden_units,
            num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0  # Dropout only between LSTM layers
        ).to(self.device)

        # NO additional output dropout layer - dropout is already handled between LSTM layers
        self.fc = nn.Linear(hidden_units, 1).to(self.device)
        
        if self.apply_clipped_relu:
            self.final_activation = torch.nn.Hardtanh(min_val=0, max_val=1)
        else:
            self.final_activation = nn.Identity()

    def forward(self, x, h_s=None, h_c=None):
        x = x.to(self.device)
        out, (h_s, h_c) = self.lstm(x, (h_s, h_c))
        # NO output dropout - already handled between LSTM layers
        out = self.fc(out[:, -1, :])
        out = self.final_activation(out)
        return out, (h_s, h_c)
