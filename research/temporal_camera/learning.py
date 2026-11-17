"""Camera-XY-derived spatial supervision and coherent image-only augmentation."""
import torch
from torch.nn import functional as F


def xy_heatmap_loss(logits, refinement, xy, cell, sigma_mm=100.):
    grid=torch.arange(24,device=logits.device,dtype=logits.dtype)*100-1150
    xx=grid[None,None,:];yy=grid[None,:,None]
    dist=(xx-xy[:,0,None,None])**2+(yy-xy[:,1,None,None])**2
    soft=torch.softmax((-dist/(2*sigma_mm*sigma_mm)).flatten(1),1)
    logp=F.log_softmax(logits.flatten(1),1)
    nll=-(soft*logp).sum(1).mean()
    prob=logp.exp().reshape(-1,24,24)
    mean=torch.stack(((prob*xx).sum((1,2)),(prob*yy).sum((1,2))),1)
    offset=(xy+1200)/100-cell
    return nll+.25*F.smooth_l1_loss(mean/1200,xy/1200)+F.smooth_l1_loss(refinement,offset)


def augment_window(x,generator):
    b,channels,h,w=x.shape
    params=torch.rand((b,5),generator=generator,device=x.device)
    z=x.reshape(b,-1,3,h,w)
    gains=(.85+.30*params[:,:3])[:,None,:,None,None]
    gamma=(.8+.4*params[:,3])[:,None,None,None,None]
    z=(z*gains).clamp(0,1).pow(gamma)
    rotated=torch.flip(z,(-2,-1))
    z=torch.where((params[:,4]>.5)[:,None,None,None,None],rotated,z)
    return z.reshape_as(x)
