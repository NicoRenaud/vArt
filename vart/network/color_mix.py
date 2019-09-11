import torch
from torch import nn

class ColorMixer(nn.Module):

    def __init__(self,ncenter,init_color=None,mix='mean'):
        super(ColorMixer,self).__init__()
        self.ncenter = ncenter

        if init_color is None:
            init_color = torch.rand(self.ncenter,3)*255

        self.color = nn.Parameter(init_color)
        self.color.requires_grad = True
        self.mix = mix

    def forward(self,x):

        if self.mix == 'mean':
            x = x[...,None] * self.color[None,...]
            return x.mean(1)
        elif self.mix == 'sum':
            x = x[...,None] * self.color[None,...]
            #return x.max(1)[0]
            return x.sum(1)
        elif self.mix == 'max':
            return self.color[x.argmax(1)]
