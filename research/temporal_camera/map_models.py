"""Reuse the accepted CameraXY RGB/map weights; temporal residuals are XY-only."""
import torch
from torch import nn
from torch.nn import functional as F


class MapCameraXYNet(nn.Module):
    def __init__(self, checkpoint, route='cnn', window=1, single_frame_adjustment=False):
        super().__init__()
        if tuple(checkpoint['targets'])!=('camera_x_mm','camera_y_mm') or not checkpoint['orientations_collapsed']:
            raise ValueError('Parent is not XY-only')
        self.route,self.window=route,window
        self.single_frame_adjustment=single_frame_adjustment and route=='cnn'
        layers=[]
        for a,b in ((3,16),(16,24),(24,32),(32,48)):
            layers.extend([nn.Conv2d(a,b,3,2,1,bias=False),nn.BatchNorm2d(b),nn.ReLU()])
        self.rgb=nn.Sequential(*layers,nn.AvgPool2d(2),nn.Flatten(),nn.Linear(720,64),nn.ReLU(),nn.Linear(64,64))
        self.centers=len(checkpoint['centers_mm']);self.views=checkpoint['reference_views']
        self.register_buffer('reference',torch.zeros_like(checkpoint['state']['reference']))
        self.load_state_dict(checkpoint['state'],strict=True)
        for p in self.rgb.parameters():p.requires_grad_(False)
        if route=='gru':self.temporal=nn.GRU(64,64,batch_first=True)
        elif route=='tcn':
            self.temporal=nn.ModuleList([nn.Conv2d(64,64,(1,3),dilation=(1,d)) for d in (1,2,4)])
        elif route!='cnn':raise ValueError('Unknown route')
        if route!='cnn' or self.single_frame_adjustment:
            self.correction=nn.Linear(64,64)
            nn.init.zeros_(self.correction.weight);nn.init.zeros_(self.correction.bias)
        self.rgb.eval()

    def train(self,mode=True):
        super().train(mode);self.rgb.eval()
        return self

    def image_embedding(self,rgb):
        return F.normalize(self.rgb(rgb),dim=-1)

    def score_features(self,features):
        last=features[:,-1]
        if self.single_frame_adjustment:
            last=last+.1*self.correction(last)
        elif self.route=='gru':
            context=self.temporal(features)[0][:,-1]
            last=last+.1*self.correction(context)
        elif self.route=='tcn':
            z=features.transpose(1,2).unsqueeze(2)
            for conv,d in zip(self.temporal,(1,2,4)):
                z=z+F.relu(conv(F.pad(z,(2*d,0,0,0))))
            last=last+.1*self.correction(z[:,:,0,-1])
        scores=F.linear(F.normalize(last,dim=-1),self.reference)
        return scores.reshape(-1,self.centers,self.views).amax(-1)

    def forward(self,rgb_window):
        b,_,h,w=rgb_window.shape
        z=self.image_embedding(rgb_window.reshape(b*self.window,3,h,w)).reshape(b,self.window,64)
        return self.score_features(z)
