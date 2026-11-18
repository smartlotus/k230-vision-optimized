"""Fixed90x160 spatial encoder avoids the verified GAP/activation collapse."""
from torch import nn
from models import CameraXYNet as Baseline


class SpatialEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        for a,b in ((3,16),(16,24),(24,32)):
            layers.extend([nn.Conv2d(a,b,3,2,1,bias=False),nn.BatchNorm2d(b),nn.ReLU()])
        self.conv=nn.Sequential(*layers)
        self.pool=nn.AvgPool2d(4,4)
        self.fc=nn.Linear(32*3*5,64)

    def forward(self,x):
        return self.fc(self.pool(self.conv(x)).flatten(1))


class CameraXYNet(Baseline):
    def __init__(self,route='cnn',window=1,grid=24,features=64):
        super().__init__(route,window,grid,features)
        if features!=64:raise ValueError('Frozen spatial encoder features64')
        self.encoder=SpatialEncoder()
