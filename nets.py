'''
Set of neural networks and various custom activations. 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from utils import rand_simplex

def init_weights(net, init_dict, gain=1, input_class=None):
    def init_func(m):
        if input_class is None or type(m) == input_class:
            for key, value in init_dict.items():
                param = getattr(m, key, None)
                if param is not None:
                    if value == 'normal':
                        nn.init.normal_(param.data, 0.0, gain)
                    elif value == 'xavier':
                        nn.init.xavier_normal_(param.data, gain=gain)
                    elif value == 'kaiming':
                        nn.init.kaiming_normal_(param.data, a=0, mode='fan_in')
                    elif value == 'orthogonal':
                        nn.init.orthogonal_(param.data, gain=gain)
                    elif value == 'uniform':
                        nn.init.uniform_(param.data)
                    elif value == 'zeros':
                        nn.init.zeros_(param.data)
                    elif value == 'very_small':
                        nn.init.constant_(param.data, 1e-3*gain)
                    elif value == 'ones':
                        nn.init.constant_(param.data, 1)
                    elif value == 'xavier1D':
                        nn.init.normal_(param.data, 0.0, gain/param.numel().sqrt())
                    elif value == 'identity':
                        nn.init.eye_(param.data)
                    else:
                        raise NotImplementedError('initialization method [%s] is not implemented' % value)
    net.apply(init_func)

class relu2(nn.Module):
    def __init__(self,order=2):
        super(relu2,self).__init__()
        self.a = nn.Parameter(torch.ones(1))
        self.order = order

    def forward(self,x):
        return F.relu(x)**(self.order)

class SMLP(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_size, act=nn.ReLU(), bn=True):
        super(SMLP, self).__init__()

        self.act = act

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        mid_list = []
        for i in range(layers):
           mid_list += [nn.Linear(hidden_size,hidden_size), nn.BatchNorm1d(hidden_size), act]
        self.mid = nn.Sequential(*mid_list)
        self.out = nn.Linear(hidden_size, out_size, bias=True)

    def forward(self,x):
        out = self.fc1(x)
        out = self.bn(out)
        out = self.act(out)
        out = self.mid(out)
        out = self.out(out)
        return F.relu(out)

class SqrtReLUBeta(nn.Module):
    def __init__(self):
        super(SqrtReLUBeta, self).__init__()
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return (torch.sqrt(self.beta.to(x.device)**2 + x ** 2) + x) / 2

class MaxElementMult(nn.Module):
    '''
    Building block of the dMNN.
    Elemntwise multiplication with max activation.
    '''
    def __init__(self, in_size, out_size):
        super(MaxElementMult, self).__init__()
        a = torch.rand(out_size, in_size)#.round()
        self.in_size = in_size
        self.W = nn.Parameter(a)

    def forward(self, x):
        return torch.max(self.W * x.unsqueeze(1), -1)[0]

    def clamp(self):
        self.W.data.clamp_(0)

    def norm(self):
        return self.W.sum()

class dMNN(nn.Module):
    '''
    Class specifying dMNN with parameters of width depth and input dimension.
    '''
    def __init__(self, input_size, width, depth, cond_size=0, cond_width=0):
        super(dMNN, self).__init__()

        layers = []

        self.W0 = MaxElementMult(input_size, width)

        for _ in range(depth-1):
            Wi = MaxElementMult(width, width)
            layers.append(Wi)

        self.Wi = nn.Sequential(*layers)

    def forward(self, x, cond=None):
        assert len(x.shape) == 2, 'We will unsqueeze it here.'

        out = self.W0(x)
        out = self.Wi(out)

        return out.mean(-1, keepdims=True)

    def norm(self):
        n = self.W0.norm()
        for block in self.Wi:
            n += block.norm()
        return n

    def clamp(self):
        self.W0.clamp()
        for block in self.Wi:
            block.clamp()

class ICNN(nn.Module):
    def __init__(self, input_size, width, depth, 
            cond_size=0, 
            cond_width=0, 
            fn0=relu2(order=2), 
            fn=nn.LeakyReLU(), 
            fnu=nn.LeakyReLU()):

        super(ICNN, self).__init__()

        self.fn0 = fn0
        self.fn = fn
        self.cond_size = cond_size

        self.fc0 = nn.Linear(input_size,width,bias=True)

        if cond_size > 0:
            self.uc0   = nn.Linear(cond_size, cond_width, bias=True)
            self.cc0   = nn.Linear(cond_size, width, bias=False)
            mid_list   = [PICNN_block(input_size,width,width,cond_width,fn,fnu) for i in range(depth-1)]
            mid_list.append(PICNN_block(input_size,width,1,cond_width,nn.Softplus(),fnu))
        else:
            mid_list = [ICNN_block(input_size,width,fn) for i in range(depth-1)]
            self.out_z = nn.Linear(width, 1, bias=False)
            self.out_x = nn.Linear(input_size, 1, bias=True)

        self.mid = nn.Sequential(*mid_list)
        init_weights(self, {'weight': 'orthogonal', 'bias': 'zeros'}, gain=1)

    def forward(self, x, cond=None):
        z0 = self.fc0(x)

        if self.cond_size > 0:
            u0 = self.uc0(cond)
            c0 = self.cc0(cond)
            z0 = self.fn0(z0 + c0)
            _, z, _ = self.mid((x, z0, u0))
            return z
        else:
            z0 = self.fn0(z0)
            _, z = self.mid((x,z0))
            out = (self.out_x(x) + self.out_z(z))
            return out

    def clamp(self):
        if self.cond_size == 0:
            self.out_z.weight.data.clamp_(0)
        for block in self.mid:
            block.clamp()

class ICNN_block(nn.Module):
    def __init__(self, x_size, zi_size, fn):
        super(ICNN_block, self).__init__()
        self.lin_x = nn.Linear(x_size, zi_size, bias=True)
        self.lin_z = nn.Linear(zi_size, zi_size, bias=False)
        self.fn = fn

    def forward(self, input_):
        x = input_[0]
        z = input_[1]
        out = self.fn(self.lin_x(x) + self.lin_z(z))
        return (x, out)

    def clamp(self):
        self.lin_z.weight.data.clamp_(0)


class PICNN_block(nn.Module):
    def __init__(self, x_size, zi_size, zout_size, ui_size, fn, fnu):
        super(PICNN_block, self).__init__()

        self.lin_u_hat = nn.Linear(ui_size, ui_size, bias=True)

        self.lin_u  = nn.Linear(ui_size, zout_size, bias=True)
        self.lin_uz = nn.Linear(ui_size, zi_size, bias=True)
        self.lin_ux = nn.Linear(ui_size, x_size,  bias=True)

        self.lin_x  = nn.Linear(x_size,  zout_size, bias=False)
        self.lin_z  = nn.Linear(zi_size, zout_size, bias=False)

        self.fn  = fn
        self.fnu = fnu


    def forward(self, input_):

        x = input_[0]
        z = input_[1]
        u = input_[2]

        u1  = self.fnu( self.lin_u_hat( u ) ) 

        pos = self.lin_z( z * F.relu( self.lin_uz( u ) ) )
        wx  = self.lin_x( x * self.lin_ux( u ) )
        wu  = self.lin_u( u )
        z1 = pos + wx + wu

        if self.fn:
            z1  = self.fn( z1 ) 

        return (x, z1, u1)

    def clamp(self):
        self.lin_z.weight.data.clamp_(0)

class L2Proj(nn.Module):
    def __init__ (self):
        super(L2Proj, self).__init__()
    def forward(self, x):
        if torch.norm(x) > 1:
            return x/torch.norm(x)
        else:
            return x

