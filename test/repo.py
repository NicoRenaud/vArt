import itertools
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from PIL import Image
import numpy as np
from vart.network.rbf import RBF
from vart.network.color_mix import ColorMixer
from vart.solver.torch_utils import Loss, DataSet

import matplotlib.pyplot as plt


def paint(net,size,dsamp=10,save=None,show=True):

    x = list(range(0,size[1],dsamp))
    y = list(range(0,size[0],dsamp))

    grid = torch.tensor(list(itertools.product(x,y))).float()
    grid = torch.index_select(grid,1,torch.LongTensor([1,0]))

    vals = net(grid).reshape(len(x),len(y),3).detach().numpy()
    vals = (vals).astype('uint8')

    new_im = Image.fromarray(vals)

    if show:
        new_im.show()

    if save is not None:
        new_im.save(save)
    return new_im


# define the network
class RepoNet(nn.Module):

    def __init__(self,ncenter,size,sigma=200):
        super(RepoNet,self).__init__()
        self.ncenter = ncenter
        self.center = torch.cat((torch.randint(0,size[0],size=(ncenter,1)),
                                 torch.randint(0,size[1],size=(ncenter,1))),
                                 dim=1).float()
        

        self.rbf = RBF(2,ncenter,self.center, sigma = sigma,Pmax=8)
        self.mix = ColorMixer(self.ncenter,mix='max')

    def forward(self,x):
        x = self.rbf(x)
        return self.mix(x)
        
# load image 
fname = '../images/pearl.jpg'
# fname = '../images/sunflower.jpg'
# fname = '../images/almandbloem.jpg'

im = Image.open(fname)
px = im.load()

# define the network
ncenter = 500
net = RepoNet(ncenter,im.size)

# initial points
npts = 1000
pts = torch.cat((torch.randint(0,im.size[0],size=(npts,1)),
                 torch.randint(0,im.size[1],size=(npts,1))),
                 dim=1).float()

data = DataSet(pts,px)
bsize = 100
dataloader = DataLoader(data,batch_size=bsize)

# for i in range(ncenter):
#     net.mix.color.data[i,:] = torch.tensor(px[int(net.center[i,0].item()),int(net.center[i,1].item())])


criterion = Loss()
#optimizer = optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
optimizer = optim.Adam(net.parameters(),lr=5)


# paint init pic
init_im = paint(net,im.size)



# train 
nepoch = 100
isave = 0
ploss = []
for n in range(nepoch):
    
    cl = 0
    for d in dataloader:

        pts, cols = d 
        
        optimizer.zero_grad()
        out = net(pts)
        loss = criterion(out,cols)
        cl += loss
        loss.backward()
        optimizer.step()


        net.rbf.centers.data += (-1+2*(torch.rand(net.rbf.ncenter,2)))*10

        net.rbf.centers.data[:,0] = net.rbf.centers.data[:,0]%im.size[0]
        net.rbf.centers.data[:,1] = net.rbf.centers.data[:,1]%im.size[1]

    if n%10 == 0:
        fname = 'rbf_girl_%03d.png' %isave
        paint(net,im.size,save=fname,show=False)
        isave += 1

        # dataloader.dataset.data = torch.cat((torch.randint(0,im.size[0],size=(npts,1)),
        #          torch.randint(0,im.size[1],size=(npts,1))),
        #          dim=1).float()

    ploss.append(cl.item())
    print(n,cl.item())

plt.plot(ploss)
plt.show()


new_im = paint(net,im.size,dsamp=5,save='rbf_girl.png')

