'''
Scripts for computing the generative model for synthetic data 
from known copulas. 
'''
import os

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader, random_split

from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pl_bolts.callbacks import PrintTableMetricsCallback

from datasets import ASLProcess, SLProcess, ASL, SL, Ozone
from gen import GenerativePickandsModule
from pickands import ConditionalPickandsModule
from nets import SMLP, MaxLinear
from utils import init_estimators, CFGEstimator, rand_simplex

import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

seed_everything(777)


# Trainer parameters
trainer_params = {
        'gpus' : 1,
        'check_val_every_n_epoch' : 500, 
        'max_epochs' : 5000, 
        'progress_bar_refresh_rate' : 0
        }


def get_data(d, alpha, data='sl'):

    n_data = 1000

    def pickands_sl(w, alpha=alpha):
        if alpha == 0:
            return w.max(1)[0]

        return torch.sum(w ** (1 / alpha), dim=1) ** alpha

    alpha  = torch.ones(1) * alpha
    alphas = torch.stack([0.5 * torch.ones_like(alpha), alpha], 0)
    thetas = torch.rand(d)
    thetas = torch.stack((thetas, 1 - thetas), dim=0)

    def pickands_asl(w, alphas=alphas, thetas=thetas):
        wtheta = w.unsqueeze(1) * thetas
        out_alpha_pos = torch.sum(wtheta ** (1. / alphas), dim=2, keepdim=True) ** alphas
        out_alpha_zero = torch.max(wtheta, dim=2, keepdim=True)[0]
        return torch.sum(torch.where(alphas > 0, out_alpha_pos, out_alpha_zero), dim=1).squeeze()

    x = torch.linspace(0, 1-1e-8)
    x_ = torch.stack((x, 1 - x), dim=1)
    if data == 'sl':
        dataset = SL(d, n_data, alpha)
        dataset_test = SL(d, n_data, alpha)
        pickands = pickands_sl
    elif data == 'asl':
        dataset = ASL(alphas, thetas, n_data)
        dataset_test = ASL(alphas, thetas, n_data)
        pickands = pickands_asl

    train_loader = DataLoader(dataset, batch_size=n_data, shuffle=False)
    test_loader  = DataLoader(dataset_test, batch_size=n_data, shuffle=False)
    return pickands, train_loader, test_loader

# Data parameters

def train(alpha, d, data, log=True):
    print(type(alpha))
    print(type(d))
    if type(d) == int:
        iter_alpha = True
        iter_ = alpha
        assert type(alpha) == np.ndarray
    elif type(alpha) == float:
        iter_alpha = False
        iter_ = d
        assert type(d) == np.ndarray
    else:
        raise TypeError 

    for idx, i in enumerate(iter_):
        if iter_alpha:
            alpha = i
        else:
            d = i 

        d = int(d)

        # Architecture parameters
        depth = 2 
        width = 256
        l = 128

        lr = 1e-3

        pickands, train_loader, test_loader = get_data(d, alpha, data)
        dataset = train_loader.dataset

        # init training module
        net = SMLP(l, width, depth, d)
        gpm = GenerativePickandsModule(net=net, cov=None, pickands=pickands, lr=lr, latent_size=l, d=d, use_swa=False)
        ex_name = "d={}_a={}_data={}".format(d, alpha, data)
        v = 2

        # train model
        if log:
            logger = CSVLogger("logs_generative", name=ex_name, version=v)
            checkpoint_callback = ModelCheckpoint(
                    monitor = 'mse',
                    dirpath = os.path.join(logger.log_dir, 'checkpoints')
                    )
            trainer = Trainer(logger=logger,callbacks=[checkpoint_callback],**trainer_params)
        else:
            trainer = Trainer(**trainer_params)

        stats = train_dmax(train_loader, test_loader, d, pickands)

        print('DMNN Finished for {}'.format(ex_name))
        print(stats)

        trainer.fit(gpm, train_loader, test_loader)

        with open('logs_generative/{}/version_{}/dmnn.pkl'.format(ex_name, v), 'wb') as f:
            import pickle
            pickle.dump(stats, f)

def train_dmax(tl, vl, d, a):

    WIDTH = 512
    DEPTH = 1

    n_simplex = 1000
    
    model_params = {
            'lr' : 1e-2,
            'survival' : False,
            'use_swa'  : False
            }


    # Architecture parameters
    arch_params = {
            'input_size' : d,
            'depth' : DEPTH,  # 3
            'width' : WIDTH,# 24
            'cond_size' : 0, 
            'cond_width': 0,
            }


    def sample_ml(net, N=10, n_samples=1000):
        assert N < net.W0.W.data.shape[0]
        exp = np.sort(np.random.exponential(1, (n_samples, N)), -1)
        pp  = torch.tensor(1 / exp.cumsum(-1))
        spec = net.W0.W.data / net.W0.W.data.sum(-1, keepdims=True)
        inds = np.random.randint(0, net.W0.W.shape[0], (n_samples, N))

        samples = (pp.unsqueeze(-1) * spec[inds, :]).max(1)[0]
        return samples.detach()

    print('alpha = ', a)
    net = MaxLinear(**arch_params)

    # init training module
    cpm = ConditionalPickandsModule(net, **model_params)
    cpm.val_dataloader = vl

    # train model
    trainer = Trainer(**trainer_params)
    trainer.fit(cpm, tl, vl)

    if trainer_params['gpus'] == 1:
        net.to('cuda:0') # keep it on the gpu (super hacky)

    samps = sample_ml(cpm.net)
    real = vl.dataset[:1000][0]

    cfg_est  = CFGEstimator(samps)
    cfg_real = CFGEstimator(real)
    w = rand_simplex(n_simplex, d)

    ce = cfg_est(w)
    cr_cfg = cfg_real(w)
    cr = a(w)

    import torch.nn.functional as F

    mse = F.mse_loss(ce, cr)
    std = ((ce - cr)**2).std()
    mse_real = F.mse_loss(cr_cfg, cr)
    std_real = ((cr - cr_cfg)**2).std()
    rel = (1 - ce/cr).abs().mean()
    rel_real = (1 - cr_cfg/cr).abs().mean()
    stats = {'dmnn_mse' : mse, 'dmnn_std' : std, 'dmnn_rel' : rel}
    return stats


# Train ASL for different alpha
train(alpha=np.linspace(0,1,9), d=225, data='asl',log=True) 

# Train SL for different alpha
train(alpha=np.linspace(0,1,9), d=225, data='sl',log=True) 

# Train SL for different dimensions
train(alpha=.5, d=np.array([ 8, np.sqrt(128), 16, 28, 32 ]) **2, data='sl', log=True)

# Train ASL for different dimensions
train(alpha=.5, d=np.array([ 8, np.sqrt(128), 16, 28, 32 ]) **2, data='asl', log=True)
