import torch
import torch.nn as nn


class TasteNetChoice(nn.Module):
    """TasteNet-MNL: tastes b = TasteNet(z), utility v = -CO + b_TT * TT + b_POI * POI (no ASC)."""

    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()
        self.taste = TasteNet(layer_sizes, activation)
        self.util = Utility()

    def forward(self, z, x):
        return self.util(x, self.taste(z))      # (N, J)


class TasteNet(nn.Module):
    """z (N, D) -> tastes (N, 2) = [b_TT, b_POI]: one hidden ReLU layer and a linear output layer.
    z is standardized with the training mean / SD stored in the buffers z_mean / z_std."""

    def __init__(self, layer_sizes, activation='relu'):
        super().__init__()
        self.seq = nn.Sequential()
        for i, (n_in, n_out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.seq.add_module(f'L{i + 1}', nn.Linear(n_in, n_out))
            if i < len(layer_sizes) - 2:
                self.seq.add_module(f'A{i + 1}', {'tanh': nn.Tanh(), 'relu': nn.ReLU()}[activation])
        self.register_buffer('z_mean', torch.zeros(layer_sizes[0]))
        self.register_buffer('z_std', torch.ones(layer_sizes[0]))

    def forward(self, z):
        return self.seq((z - self.z_mean) / self.z_std)


class Utility(nn.Module):
    """x (N, K, J) with rows [CO, TT, POI]; b (N, 2). Cost coefficient fixed at -1."""

    def forward(self, x, b):
        return -x[:, 0, :] + b[:, [0]] * x[:, 1, :] + b[:, [1]] * x[:, 2, :]
