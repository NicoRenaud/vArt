import itertools
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PIL import Image, ImageFilter
import numpy as np

from vart.sampler.metropolis import Metropolis

from vart.network.rbf import RBF
from vart.network.color_mix import ColorMixer
from vart.solver.torch_utils import Loss, DataSet

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.ndimage.filters import gaussian_filter

class Density(object):

    def __init__(self,data):
        self.data = data

    def __call__(self,pts):
        pts = pts.int().numpy().tolist()
        return torch.tensor([self.data[i,j] for i,j in pts])

def Gaussian(pts):
    r = (pts**2).sum(1)
    return torch.exp(-r) 

# define the network
class RepoNet(nn.Module):

    def __init__(self,ncenter,size,sigma=200):
        super(RepoNet,self).__init__()
        self.ncenter = ncenter
        self.center = torch.cat((torch.randint(0,size[0],size=(ncenter,1)),
                                 torch.randint(0,size[1],size=(ncenter,1))),
                                 dim=1).float()
        

        self.rbf = RBF(2,ncenter,self.center, sigma = sigma,Pmax=8)
        self.fc = nn.Linear(self.ncenter,1,bias=False)

    def forward(self,x):
        x = self.rbf(x)
        return self.fc(x)


#load
fname = '../images/pearl.jpg'
im = Image.open(fname)

# resize
size = (int(s/10) for s in im.size)
im = im.resize(size)

# filter
im = im.filter(ImageFilter.SMOOTH_MORE)

# load
imr,img,imb = im.split()
rsize = imr.size

fig = plt.figure()
ax = fig.gca(projection='3d')
imrd = np.array(imr.getdata()).reshape(rsize[1],rsize[0])

imrd = gaussian_filter(imrd,sigma=10)

x = np.linspace(0,rsize[0]/10,rsize[0])
y = np.linspace(0,rsize[1]/10,rsize[1])
xx,yy = np.meshgrid(x,y)
ax.plot_surface(xx,yy,imrd)
plt.show()

# define red density
red_pdf = Density(imrd.T)

# sampler
sampler = Metropolis(nwalkers=100, 
                     nstep=1000, 
                     ndim = 2,
                     step_size = 0.5 )

pos = sampler.generate(red_pdf,rsize, ntherm=-1,init='uniform')

plt.scatter(pos[:,0],pos[:,1])
plt.show()

