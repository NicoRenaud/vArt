import torch
from torch import nn
import torch.nn.functional as F
from math import pi as PI

def pairwise_distance(x,y):

    xn = (x**2).sum(1).view(-1,1)
    yn = (y**2).sum(1).view(1,-1)
    return xn + yn + 2.*x.mm(y.transpose(0,1))

########################################################################################################

class RBF(nn.Module):

    def __init__(self,
                input_features,
                output_features,
                centers,
                kernel='gaussian',
                sigma = 1.0,
                Pmax= 1 ):

        '''Radial Basis Function Layer in N dimension

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
            sigma : strategy to get the sigma
            opt_sigma : optmize the std or not
        '''

        super(RBF,self).__init__()

        # register dimension
        self.input_features = input_features
        self.output_features = output_features

        # make the centers optmizable or not
        self.centers = nn.Parameter(torch.Tensor(centers))
        self.ncenter = len(self.centers)
        self.centers.requires_grad = True
        self.kernel = kernel

        # get the standard deviations
        self.sigma = nn.Parameter(sigma*torch.rand(self.centers.shape))
        self.sigma.requires_grad = True

        # get the covariance matrix and its inverse
        self.invCov = self.invCovMat(self.sigma)

        self.P = nn.Parameter(torch.randint(0,Pmax,size=(self.ncenter,)).float())
        self.P.requires_grad = True

        # GET THE DENOMINATOR
        #self.detS = self.denom(self.sigma,self.input_features)

        # get the scaled determinant of the cov matrices
        # self.detS = (self.sigma**2).prod(1).view(-1,1)
        # k = (2.*PI)**self.input_features
        # self.detS = torch.sqrt( k*self.detS )

    @staticmethod
    def invCovMat(sigma):
        s2 = sigma**2
        I = torch.eye(sigma.size(1))
        cov = s2.unsqueeze(2).expand(*s2.size(),s2.size(1))
        return torch.inverse(cov * I)

    @staticmethod
    def denom(sigma,dim):
        out = (sigma**2).prod(1).view(-1,1)
        k = (2.*PI)**dim
        return torch.sqrt( k*out )
        
    def forward(self,input):
        '''Compute the output of the RBF layer'''

        if self.kernel == 'gaussian':
            return self._gaussian_kernel(input)

        else:
            raise ValueError('Kernel not recognized')

    def _gaussian_kernel(self,input):

        if self.sigma.requires_grad:
            self.invCov = self.invCovMat(self.sigma)
            self.detS = self.denom(self.sigma,self.input_features)

        # get the distancese of each point to each RBF center
        # (Nbatch,Nrbf,Ndim)
        delta =  (input[:,None,:] - self.centers[None,...])

        # Compute (INPUT-MU).T x Sigma^-1 * (INPUT-MU)-> (Nbatch,Nrbf)
        X = ( torch.matmul(delta.unsqueeze(2),self.invCov).squeeze(2) * delta ).sum(2)
        
        # divide by the determinant of the cov mat
        X = torch.exp(-0.5*X**self.P)

        return X



