'''
Pytorch-Lightning Module for sampling from a spectral measure
and learning the Pickands dependence function using the expectation.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.distributions.weibull import Weibull
from torch.optim.swa_utils import AveragedModel, SWALR

import numpy as np

import pytorch_lightning as pl

from utils import rand_simplex, CFGEstimator

from scipy.stats import invweibull

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import math

from scipy import stats

class GenerativePickandsModule(pl.LightningModule):

    def __init__(self, pickands, cov: nn.Parameter,  
            net: nn.Module = None,  
            lr: float = 1e-4, 
            survival: bool = False, 
            d: int = 2, 
            latent_size: int = 2, 
            loss: str = 'mle',
            use_swa: bool = False):

        super(GenerativePickandsModule, self).__init__()
        self.loss = loss
        self.cov = cov 
        self.net = net 
        self.use_swa = use_swa
        if self.use_swa:
            self.swa_net = AveragedModel(self.net)
        self.lr  = lr
        self.pickands = pickands
        self.survival = survival
        self.d = d
        self.latent_size = latent_size
        self.weibull = Weibull(torch.tensor([1.0]), torch.tensor([1.0]))
        self.save_hyperparameters()


    def training_step(self, batch, batch_idx):
        loss = 0
        if self.loss == 'segers':
            loss = self.segers_loss(batch, batch_idx)
        elif self.loss == 'dist':
            loss =  self.distance_loss(batch, batch_idx)
        elif self.loss == 'mle':
            loss =  self.mle_loss(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def segers_loss(self, batch, batch_idx):
        n_sample  = 1000
        n_simplex = 1000

        w = rand_simplex(n_simplex, self.d) # n_simplex x d 
        noise = torch.randn(n_sample, self.latent_size) 
        vec = self.net(noise)# + 1 # n_sample x d
        vec_mean = vec.mean(0)
        reg = F.mse_loss(vec_mean, torch.ones_like(vec_mean))

        m = ( w.unsqueeze(1) * (vec) ).max(2)[0]
        if self.d < 10:
            loss = F.mse_loss(self.pickands(w), m.mean(1)) + reg
        else:
            loss = F.mse_loss(self.pickands(w).log(), m.mean(1).log()) + reg

        return loss

    def distance_loss(self, batch, batch_idx):
        n_sample  = 1000
        n_simplex = 1000

        w = rand_simplex(n_simplex, self.d) # n_simplex x d 

        a_true = self.pickands(w)

        samps, _ = self.sample(n_sample)
        a_hat = self.learned_CFG(samps, w)

        return F.mse_loss(a_hat, a_true)

    def mle_loss(self, batch, batch_idx):
        loss = self.exp_mle_iid(batch)
        return loss

    def validation_step(self, batch, batch_idx):
        real = batch[0]

        n_sample  = real.shape[0] 
        n_simplex = 1000

        all_p = []
        n_p = 10

        nv, vec = self.sample(n_sample)
        cfg_est  = CFGEstimator(torch.Tensor(nv))
        cfg_real = CFGEstimator(real)
        w = rand_simplex(n_simplex, self.d) 

        ce = cfg_est(w)
        cr_cfg = cfg_real(w)
        cr = self.pickands(w)

        if self.d == 2:

            x = torch.linspace(0, 1-1e-8)
            x_ = torch.stack((x, 1 - x), dim=1)
            ce = cfg_est(x_)
            cr = cfg_real(x_)
            cr_cfg = cfg_real(x_)
            cr = self.pickands(x_)

            plt.plot(ce.detach().cpu().numpy(), label='sim', alpha= 0.5)
            plt.plot(cr.cpu().numpy(), label='real', alpha = 0.5)
            plt.legend()
            plt.savefig('compare_pickands.pdf')
            plt.close('all')

            nv_min = nv[nv.max(1) < 10]

            plt.scatter(nv_min[:,0], nv_min[:,1])
            plt.savefig('test_generated_f.pdf')
            plt.close('all')
            plt.scatter(vec[:,0].detach().cpu().numpy(), vec[:,1].detach().cpu().numpy())
            plt.savefig('test_generated.pdf')
            plt.close('all')

        elif self.d > 100 and int(np.sqrt(self.d))**2 == self.d:

            nv_im = nv.reshape(-1, 1, int(np.sqrt(self.d)), int(np.sqrt(self.d)))
            grid = make_grid(torch.Tensor(nv_im[:64]), scale_each=True, normalize=True)
            plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1,2,0)))
            plt.savefig('sampled_imgs.png')
            plt.close('all')
            real_im = real.reshape(-1, 1, int(np.sqrt(self.d)), int(np.sqrt(self.d)))
            grid = make_grid(torch.Tensor(real_im[:64]), scale_each=True, normalize=True)
            plt.imshow(np.transpose(grid.detach().cpu().numpy(), (1,2,0)))
            plt.savefig('real_imgs.png')
            plt.close('all')


        mse = F.mse_loss(ce, cr)
        std = ((ce - cr)**2).std()
        mse_real = F.mse_loss(cr_cfg, cr)
        std_real = ((cr - cr_cfg)**2).std()
        rel = (1 - ce/cr).abs().mean()
        rel_real = (1 - cr_cfg/cr).abs().mean()
        print('MSE ', mse.item())
        print('Rel ', rel.item())
        self.log('mse', mse.item())
        self.log('std', std.item())
        self.log('rel', rel.item())
        self.log('mse_real', mse_real.item())
        self.log('std_real', std_real.item())
        self.log('real_real', rel_real.item())

        return mse_real

    def sample(self, n_sample):

        self.net.eval()

        tries = 10
        cache = np.zeros((n_sample, tries, self.d))

        for t in range(tries):

            if self.cov is not None:
                if self.cov.shape[0] == self.cov.shape[1]:
                    c = self.cov.T @ self.cov
                else:
                    L = torch.zeros(self.d, self.d)
                    L[0,0] = self.cov[0]
                    L[1,0] = self.cov[1]
                    L[1,1] = self.cov[2]

                    c = L.T @ L

                noise = torch.rand(n_sample, self.d)
                vec = F.relu(1 + noise @ c) 

            elif self.net is not None:
                noise = torch.randn(n_sample, self.latent_size) 
                if self.use_swa:
                    vec = F.relu( self.swa_net(noise) )# n_sample x d
                else:
                    vec = F.relu( self.net(noise) ) # n_sample x d

            fre = invweibull.rvs(1, size=n_sample)
            nv  = fre.reshape(-1,1) * vec.detach().cpu().numpy()
            cache[:, t, :] = nv.copy()

        nv = cache.max(1)
        return nv, vec

    def exp_mle_iid(self, batch):

        y, rank = batch
        bs, d   = y.shape

        neg_log_rank = -rank.log()
        t = rand_simplex(bs, d) # bs x d
        zt, _ = (neg_log_rank / t).min(1, keepdims=True)

        # calculate a 
        n_spectral = 1000
        noise = torch.randn(n_spectral, self.latent_size)
        vec = self.net(noise)
        vec_mean = vec.mean(0)
        m = ( t.unsqueeze(1) * (vec) ).max(2)[0]

        a = m.mean(1).unsqueeze(1)

        reg = F.mse_loss(vec_mean, torch.ones_like(vec_mean))

        loss = -(a.log() - (zt * a)).mean() + reg
        return loss

    def learned_CFG(self, samples, w):
        n_samples = samples.shape[0]
        logF = -1 / samples
        xi = (-logF / w.unsqueeze(1)).min(dim=1)[0]
        hA = (-xi.log().mean(dim=1) - np.euler_gamma).exp()
        return hA

    def on_after_backward(self):
        if self.use_swa: 
            self.swa_net.update_parameters(self.net)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.5, 0.99))
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99998) 
        if self.use_swa: 
            scheduler= SWALR(optimizer, swa_lr=0.005)
        return [optimizer], [scheduler]


