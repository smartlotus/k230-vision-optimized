"""Small CameraXY-only models; packed RGB window, fixed T, no yaw head."""
import torch
from torch import nn
from torch.nn import functional as F


class Encoder(nn.Module):
    def __init__(self, features=64):
        super().__init__()
        layers = [nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU()]
        for a, b in [(16, 24), (24, 32), (32, 48), (48, 64)]:
            layers.extend([nn.Conv2d(a, a, 3, 2, 1, groups=a), nn.ReLU(),
                           nn.Conv2d(a, b, 1), nn.ReLU()])
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(64, features)

    def forward(self, x):
        return self.fc(self.conv(x).mean((2, 3)))


class CameraXYNet(nn.Module):
    def __init__(self, route='cnn', window=1, grid=24, features=64):
        super().__init__()
        if route not in ('cnn', 'gru', 'tcn') or window < 1:
            raise ValueError('Invalid fixed route/window')
        self.route, self.window, self.grid = route, window, grid
        self.encoder = Encoder(features)
        if route == 'gru':
            self.gru = nn.GRU(features, features, batch_first=True)
        elif route == 'tcn':
            self.temporal = nn.ModuleList([nn.Conv2d(features, features, (1, 3),
                                                    dilation=(1, d)) for d in (1, 2, 4)])
        self.location = nn.Linear(features, grid * grid)
        self.refinement = nn.Linear(features, 2)

    def forward(self, rgb_window):
        b, _, h, w = rgb_window.shape
        z = self.encoder(rgb_window.reshape(b * self.window, 3, h, w))
        z = z.reshape(b, self.window, -1)
        if self.route == 'gru':
            z = self.gru(z)[0][:, -1]
        elif self.route == 'tcn':
            z = z.transpose(1, 2).unsqueeze(2)
            for conv, dilation in zip(self.temporal, (1, 2, 4)):
                z = z + F.relu(conv(F.pad(z, (2 * dilation, 0, 0, 0))))
            z = z[:, :, 0, -1]
        else:
            z = z[:, -1]
        return self.location(z).reshape(b, 1, self.grid, self.grid), torch.sigmoid(self.refinement(z))
