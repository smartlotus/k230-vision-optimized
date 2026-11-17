"""Fixed-window ONNX lowering for the existing one-layer CameraXY GRU."""
import torch
from torch import nn
from torch.nn import functional as F


class FixedWindowGRU(nn.Module):
    def __init__(self, gru, window):
        super().__init__()
        if (window not in (4, 8) or gru.num_layers != 1 or not gru.batch_first
                or gru.bidirectional or not gru.bias or gru.input_size != gru.hidden_size):
            raise ValueError('Only the CameraXY fixed-window inference GRU is supported')
        self.window = window
        self.state_size = gru.hidden_size
        self.register_buffer('weight_input', gru.weight_ih_l0.detach().clone())
        self.register_buffer('weight_hidden', gru.weight_hh_l0.detach().clone())
        self.register_buffer('bias_input', gru.bias_ih_l0.detach().clone())
        self.register_buffer('bias_hidden', gru.bias_hh_l0.detach().clone())

    def forward(self, features):
        step_size = self.state_size
        previous = torch.zeros_like(features[:, 0, :])
        inputs = F.linear(features, self.weight_input, self.bias_input)
        sequence = []
        for step in range(self.window):
            data = inputs[:, step, :]
            memory = F.linear(previous, self.weight_hidden, self.bias_hidden)
            reset = torch.sigmoid(data[:, :step_size] + memory[:, :step_size])
            update = torch.sigmoid(data[:, step_size:2 * step_size] + memory[:, step_size:2 * step_size])
            candidate = torch.tanh(data[:, 2 * step_size:] + reset * memory[:, 2 * step_size:])
            previous = (1. - update) * candidate + update * previous
            sequence.append(previous)
        return torch.stack(sequence, dim=1), previous.unsqueeze(0)
