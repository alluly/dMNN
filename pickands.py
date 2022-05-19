'''
Pytorch Lightning module for training a Pickands dependence function.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR

import pytorch_lightning as pl

from utils import rand_simplex

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import math

class ConditionalPickandsModule(pl.LightningModule):
    def __init__(self, 
            net: nn.Module, 
            lr: float = 1e-4, 
            survival: bool = False, 
            use_swa: bool = False, 
            loss_type: str = 'mle', 
            use_hardtanh: bool = False,
            maxes: bool = True, 
            l1: float = 1e-3):

        super(ConditionalPickandsModule, self).__init__()
        self.net = net 
        self.use_swa      = use_swa
        self.loss_type    = loss_type
        self.use_hardtanh = use_hardtanh
        if self.use_swa:
            self.swa_net = AveragedModel(self.net)
        self.lr  = lr
        self.survival = survival
        self.maxes = maxes
        self.l1 = l1

    def model_survival(self, CDF, cond=None):
        '''
        threshold : 2D vector of thresholds shape = (1, 2)
        test_set : to compute the marginal CDF on u1 and u2
        This function computes the survival prob from the trained
        Pickands copula
        '''
        assert self.survival, 'Must use a survival copula'

        # compute the model survivals
        t = (1-CDF).log()
        w = t / t.sum()
        a = self.a(w)
        survival = (t.sum() * a).exp()

        return survival


    def exp_mle_iid(self, batch):

        y, rank = batch
        bs, d   = y.shape

        if self.survival: 
            neg_log_rank = -(1-rank).log()
        else:
            neg_log_rank = -rank.log()

        t = rand_simplex(bs, d).to(y.device) # bs x d
        if self.maxes:
            zt, _ = (neg_log_rank.unsqueeze(1) / t).min(2, keepdims=True) # bs x bs
        else:
            zt, _ = (neg_log_rank.unsqueeze(1) / t).max(2, keepdims=True) # bs x bs

        # calculate a 
        a = self.a(t) # bs x 1

        if self.loss_type == 'mle':
            loss = -(a.log() - (zt * a)).mean()
        elif self.loss_type == 'cfg':
            loss = F.l1_loss(a.log(), -(zt).log().mean(1) - 0.5772)
        elif self.loss_type == 'pickands':
            loss = F.mse_loss(a, (1/zt).mean(1))
        elif self.loss_type == 'mle-pick':
            loss = -(a.log() - (zt * a)).mean() + F.l1_loss(a, (1/zt).mean(1))

        return loss

    def exp_mle(self, batch):
        y, rank, cond = batch
        bs, n_samp, d = y.shape
        cond = cond.unsqueeze(2)

        neg_log_rank = -rank.log()

        w = rand_simplex(bs * n_samp, d).reshape(bs, n_samp, d) # bs x n_samp x d

        zw, _ = (neg_log_rank / w).min(2, keepdims=True) 

        a = self.a(w, cond=cond)


        loss = -(a.log() - (zw * a)).mean()
        return loss

    def a(self, w, cond=None):
        '''
        w : simplex points batch_size x num_samples x dimensions
            or batch_size x dimensions (for iid)

        cond : conditional variable
        '''
        self.net.clamp()
        if   len(w.shape) == 2:
            bs, d = w.shape
        elif len(w.shape) == 3:
            bs, n_samp, d = w.shape

        wb     = torch.eye(d).to(w.device)
        cond_b = cond

        if cond is not None:
            # for each conditioning variable enforce boundary
            wb = wb.repeat(bs, 1, 1)
            cond_b = cond[:, 0].unsqueeze(1).repeat(1, d, 1)

        if self.use_swa:
            self.swa_net = self.swa_net.to(w.device)
            b = self.swa_net(wb, cond=cond_b)
            if self.use_hardtanh:
                a = F.hardtanh(self.swa_net(w, cond=cond) - w @ b + 1, 1/d, 1)
            else:
                a = F.relu(self.swa_net(w, cond=cond) - w @ b + 1)
        else:
            b = self.net(wb, cond=cond_b)
            a = F.relu(self.net(w, cond=cond) - w @ b + 1)

        return a

    def cdf(self, vec, cond=None):
        if cond is not None:
            pass
        pass

    def training_step(self, batch, batch_idx):
        if   len(batch) == 3:
            return self.exp_mle(batch)
        elif len(batch) == 2:
            return self.exp_mle_iid(batch) + self.l1 * self.net.norm()

    def valid_conditional(self, batch, batch_idx):

        vl = self.val_dataloader.dataloader.dataset
        y, rank, cond, w, pickands = batch
        bs, n_samp, d = y.shape
        cond = cond.unsqueeze(2)

        if d == 2:
            self.plot(batch)

        neg_log_rank = -rank.log()

        n_samp = w.shape[1]

        # make w over more samples
        zw, _ = (neg_log_rank / w).min(2, keepdims=True) # TODO: increase the number of conditioning 

        # for each conditioning variable enforce boundary
        b = self.net(torch.eye(d).repeat(bs,1,1), cond[:,0].unsqueeze(1).repeat(1,d,1))  
        a = (self.net(w, cond=cond) - w @ b + 1).squeeze(2)
        a = self.a(w, cond=cond)
        trueA = pickands

        mse = F.mse_loss(a, trueA)
        mae = F.l1_loss(a, trueA)
        pe  = (a - trueA).abs() / trueA.abs() * 100

        self.log('Pickands MSE', mse)
        self.log('Pickands MAE', mae)
        self.log('Pickands % err', pe.mean())
        self.log('Pickands % err std', pe.std())

        if self.survival:

            threshold = torch.tensor([[0.4, 0.5]])
            s_true = vl.true_survival(threshold, batch_idx)
            s_hat  = self.model_survival(threshold, cond)

        print('MSE ', mse.item())
        print('MAE ', mae.item())
        print('Percent Error ', pe.mean().item())

    def valid_iid(self, batch, batch_idx):
        vl = self.val_dataloader.dataset
        y, rank = batch
        bs, d = y.shape

        if d == 2:
            self.plot_iid(batch)

        n_samps = 1000

        b = self.net(torch.eye(d).to(y.device))
        w = rand_simplex(n_samps, d).to(y.device)

        a = self.a(w)
        a = torch.max(self.a(w), w.max(1, keepdim=True)[0])
        trueA = (torch.sum(w ** (1 / 0.5), dim=1) ** 0.5).unsqueeze(1)

        mse = F.mse_loss(a, trueA)
        mae = F.l1_loss(a, trueA)
        pe  = (a - trueA).abs() / trueA.abs() * 100

        self.log('Pickands MSE', mse)
        self.log('Pickands MAE', mae)
        self.log('Pickands % err', pe.mean())
        self.log('Pickands % err std', pe.std())

        print('mse ', mse.item())
        print('mae ', mae.item())
        print('percent error ',pe.mean().item())

        if self.survival: 

            # survival 
            threshold = 120 * torch.ones((1, d))
            CDF =  (-1 / threshold).exp() # frechet margins
            s_true, s_hat, s_mse, s_pe = self.survival_prob(CDF, vl) # this is really badly written

            self.log('s_true', s_true.item())
            self.log('s_hat' , s_hat.item())
            self.log('s_mse' , s_mse.item())
            self.log('s_pe'  , s_pe.item())

            print('True  survival {} '.format(s_true.item()))
            print('Model survival {} '.format(s_hat.item()))
            print('Survival mse   {} '.format(s_mse.item()))
            print('Survival % err {} '.format(s_pe.item()))

    def survival_prob(self, CDF, dl):
        '''
        CDF : CDF transformation of the threshold
        dl : dataloader with property 'true_survival'

        Computes errors in survival prob prediction
        '''

        s_true = dl.true_survival(CDF) # True survival
        s_hat  = self.model_survival(CDF) # model survival
        s_mse  = F.mse_loss(s_hat.squeeze(1).squeeze(0), s_true) # MSE 
        s_pe   = (s_true - s_hat.squeeze(1)).abs() / s_true.abs() * 100 # percent error

        return s_true, s_hat, s_mse, s_pe

    def validation_step(self, batch, batch_idx):
        if   len(batch) == 5:
            self.valid_conditional(batch, batch_idx)
        elif len(batch) == 2:
            self.valid_iid(batch, batch_idx)

    def plot_iid(self, batch):
        y, rank = batch

        x = torch.linspace(0, 1-1e-8)
        x_ = torch.stack((x, 1 - x), dim=1)

        plot = self.a(x_)
        plot = torch.max(self.a(x_), x_.max(1, keepdim=True)[0])

        plt.plot(x.cpu(), plot.cpu())
        plt.savefig('a_hat.pdf')
        plt.close('all')

    def plot(self, batch, num_plot=10): 

        y, rank, cond, w, pickands = batch

        cm = plt.get_cmap('inferno')

        # list of times to plot at
        times = torch.linspace(0,0.99,num_plot)

        for idx, t in enumerate(times):
            t = t.item()

            color = cm(t)
            ind   = math.floor(pickands.shape[0] * t)

            x = torch.linspace(0, 1-1e-8)
            plot = self.a(w[ind], cond=t * torch.ones((w.size(1), 1)))

            trueA = pickands[ind].cpu().detach() 

            plt.plot(x.cpu(), trueA, '--', color=color)
            plt.plot(x.cpu(), plot.cpu().detach(), color=color, label='t={:.2f}'.format(t))

        plt.title('Pickands function estimation')
        plt.xlabel('w')
        plt.ylabel('A(w)')
        plt.legend()
        plt.savefig('fcn_plot.pdf')
        plt.close('all')

    def on_after_backward(self):
        self.net.clamp()
        if self.use_swa: 
            self.swa_net.update_parameters(self.net)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)#, eps=1e-8)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)
        if self.use_swa: 
            scheduler= SWALR(optimizer, swa_lr=0.005)
        return {'optimizer' : optimizer, 'scheduler': scheduler, 'monitor' : 'train_loss'}

