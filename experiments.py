'''
This script has the experiments from the paper.
All the different experiments are defined as different functions.
'''

import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pl_bolts.callbacks import PrintTableMetricsCallback

from datasets import ASLProcess, SLProcess, ASL, SL, Ozone, IB, Commodities, Crypto, California, SP, CryptoMin
from pickands import ConditionalPickandsModule
from nets import ICNN, init_weights, MaxLinear
from utils import init_estimators, rand_simplex

import numpy as np
import itertools

from scipy.stats import genextreme as gev

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import seaborn as sns
import pandas as pd

torch.set_default_dtype(torch.float64)

WIDTH = 512
DEPTH = 1
EPOCHS = 2000
LR = 1e-2

# Trainer parameters
trainer_params = {
        'gpus' : 1,
        'check_val_every_n_epoch' : 1000, 
        'max_epochs' : EPOCHS,
        'progress_bar_refresh_rate' : 0,
        'checkpoint_callback' : False,
        'logger' : False
        }

def get_data(data_type, **kwargs):
    d = kwargs['d']
    n_samp = kwargs['n_samp']

    if data_type == 'sl_process':

        alpha = torch.linspace(0.01,1,1000)

        dataset     = SLProcess(d, n_samp, alpha=alpha, const_marginals=const_marginals)
        val_dataset = SLProcess(d, n_samp, alpha=alpha, const_marginals=const_marginals, valid=True)

    elif data_type == 'asl_process':

        alphas = torch.stack([0.5 * torch.ones_like(alpha), alpha], 1)
        thetas = torch.rand(d)
        thetas = torch.stack((thetas, 1 - thetas), dim=0)

        dataset     = ASLProcess(n_samp, theta=thetas, alpha=alphas, const_marginals=const_marginals)
        val_dataset = ASLProcess(n_samp, theta=thetas, alpha=alphas, const_marginals=const_marginals, valid=True)

    elif data_type == 'sl':
        cond_size  = 0
        cond_width = 0

        alpha = kwargs['alpha']
        
        dataset     = SL(d, n_samp, alpha)
        val_dataset = SL(d, n_samp, alpha)

    elif data_type == 'asl':
        alpha = kwargs['alpha']
        
        alphas = torch.tensor((0.5 * torch.ones_like(alpha), alpha))
        thetas = torch.rand(d)
        thetas = torch.stack((thetas, 1 - thetas), dim=0)

        dataset     = ASL(alphas, thetas, n_samp=n_samp)
        val_dataset = ASL(alphas, thetas, n_samp=n_samp)

    elif data_type == 'ozone':
        csv_files = ['data/ozone_data/sequoia_kaweah.csv', 'data/ozone_data/sequoia_ash.csv']
        #csv_files = ['data/ozone_data/sequoia_kaweah_1984.csv', 'data/ozone_data/sequoia_ash_1984.csv']
        locations = ['SEKI-LK_O3_PPB', 'SEKI-AS_O3_PPB']

        print('train')
        dataset = Ozone(csv_files, locations, month=6, year=0)
        print('Num points train: ', len(dataset))
        print('validation')
        val_dataset = Ozone(csv_files, locations, month=5, year=0)
        print('Num points validation: ', len(val_dataset))

    train_loader = DataLoader(dataset, batch_size=1000, shuffle=False)
    val_loader   = DataLoader(val_dataset, batch_size=1000, shuffle=False) 

    return train_loader, val_loader

def sample(net, N=10, n_samples=100):
    assert N < net.W0.W.data.shape[0]
    exp = np.sort(np.random.exponential(1, (n_samples, N)), -1)
    pp  = torch.tensor(1 / exp.cumsum(-1))
    spec = net.W0.W.data / net.W0.W.data.sum(-1, keepdims=True)
    inds = np.random.randint(0, net.W0.W.shape[0], (n_samples, N))

    samples = (pp.unsqueeze(-1) * spec[inds, :]).max(1)[0]
    return samples.detach().cpu()

def contour_3d(net, file_name, full_d=None, inds=None):
    from scipy.interpolate import interp2d, griddata

    w = rand_simplex(10000,3)

    if full_d is not None:
        wf = torch.zeros(10000, full_d)
        wf[:,inds] = w.clone()
        a = net(wf)
    else:
        a = net(w)

    n = 200
    x,y = np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n))
    x = x.reshape(-1)
    y = y.reshape(-1)
    ind = x + y > 1
    x[ind] = 0
    y[ind] = 0 

    grid = griddata(w[:,:2].cpu(), a.detach().cpu(), (x,y), method='linear')
    a_int = grid
    a_int[ind] = np.nan

    plt.style.use('default')
    try:
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        cnt = plt.contour(x.reshape(n,n),y.reshape(n,n),a_int.reshape(n,n), 10, cmap='magma')
        plt.clabel(cnt, inline=True, fontsize=8)
        plt.imshow(a_int.reshape(n,n), extent=[-.01,1.01,-.01,1.01], origin='lower', cmap='magma', alpha=0.3)
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close('all')
    except:
        pass

def get_all_bi_extremal_coeffs(a, d, file_name, labels=None, a_true=None, d_use=None):
    if d_use == None:
        d_use = d // 2

    # List of RGB triplets
    rgb_values = sns.color_palette("pastel", d_use * (d_use - 1) // 2)

    # Map label to RGB
    color_map = dict(zip(range(d), rgb_values))
    plt_data = []

    heatmap = np.zeros((d,d))

    mses = []
    plt.style.use('seaborn-darkgrid')

    with torch.no_grad():

        c_ind = 0 

        for s1 in range(d_use):
            for s2 in range(d_use):
                if s2 < s1:
                    continue

                one = torch.zeros(1,d)
                x = torch.linspace(0, 1-1e-8)
                x_ = torch.stack((x, 1 - x), dim=1)
                one = torch.zeros(x.shape[0],d)
                one[:,s1] = x_[:,0]
                one[:,s2] = x_[:,1]

                ex = torch.zeros(1,d)

                ex[:,s1] = 1/d
                ex[:,s2] =  1/d

                extremal = d*(a(ex))
                dep = a(one)

                heatmap[s1,s2] = extremal.detach().cpu().numpy()
                heatmap[s2,s1] = extremal.detach().cpu().numpy()
                if s1 < s2:
                    plt.plot(x.detach().cpu(), dep.detach().cpu().numpy(), linewidth=3.5, color=rgb_values[c_ind])#, color=rgb_values[s1], alpha=0.5)
                    c_ind += 1
                    if a_true is not None:
                        plt.plot(x.detach().cpu(), a_true(one).detach().cpu().numpy(), color='black')
                        if a_true(one).shape == dep.shape:
                            mses.append(F.mse_loss(a_true(one), dep).item())
                        else:
                            mses.append(F.mse_loss(a_true(one), dep.squeeze(-1)).item())

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        if len(mses) > 0:
            mse_ = np.array(mses).mean()
            mse = '{0:.2E}'.format(mse_)
            print(file_name)
            print('MSE {0:.2E}'.format(mse_))
        plt.xlabel(r'$w \in \Delta_1$', fontsize=15)
        plt.ylabel(r'$A(w, 1-w)$', fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.tight_layout()
        plt.savefig(file_name)
        plt.close('all')


def train(train_loader, 
        val_loader, 
        arch_params, 
        model_params, 
        trainer_params, 
        thresholds, 
        use_kl=False, 
        addkey=True, 
        ret_estimators=False):

    '''
    Main function for training the Pickands dependence function.
    '''

    n_runs = 1
    survival = model_params['survival']
    d = arch_params['input_size']

    np_list = ['cfg', 'naive', 'bdv']
    np_estimators = init_estimators(np_list, 
            train_loader.dataset.data, 
            survival=survival, 
            est_F=train_loader.dataset.F)

    results = { i:[] for i in range(n_runs) }

    n_thresholds = 20 # number of thresholds to consider

    for run in range(n_runs): # run the network a few times

        # init network
        net = MaxLinear(**arch_params)

        # init training module
        cpm = ConditionalPickandsModule(net, **model_params)
        cpm.val_dataloader = val_loader

        # train model
        trainer = Trainer(**trainer_params)

        icnn_start_time = datetime.now() #compute time it takes
        trainer.fit(cpm, train_loader, val_loader)
        print('Total time for training ICNN:')
        print(datetime.now() - icnn_start_time)

        # track errors for each run
        run_errs = {est : [] for est in np_list + ['nn']}
        if trainer_params['gpus'] == 1:
            net.to('cuda:0') # keep it on the gpu (super hacky)


        if survival:
            # test for various thresholds
            n_thresholds = thresholds.shape[0]
            mse_matrix = torch.zeros((n_thresholds, len(np_list)+1))

            for idx_t in range(n_thresholds):

                if type(train_loader.dataset) == Ozone \
                or type(train_loader.dataset) == IB \
                or type(train_loader.dataset) == Commodities \
                or type(train_loader.dataset) == Crypto \
                or type(train_loader.dataset) == CryptoMin \
                or type(train_loader.dataset) == California \
                or type(train_loader.dataset) == SP:

                    #t = thresholds[idx_t, :] * torch.ones((1,d))
                    t = torch.Tensor(thresholds[idx_t, :]).unsqueeze(0)

                    # get the marginal parameters
                    if trainer_params['gpus'] == 1:
                        mp = torch.Tensor(val_loader.dataset.margin_params).to('cuda:0')
                    else:
                        mp = torch.Tensor(val_loader.dataset.margin_params)

                    # normalize thresholds T_n = (T - a) / b
                    threshold_n = (t - mp[:,1]) / mp[:,2]

                    # compute marginal CDFs (using GEV params)
                    # Note that we used the convention in scipy.stats.genextreme
                    if val_loader.dataset.use_gumbel:
                        CDF = (-( - threshold_n ).exp()).exp()
                    else:
                        CDF = ( -( 1 - mp[:,0] * threshold_n )** (1 / mp[:,0]) ).exp()

                    s_est = cpm.model_survival(CDF)
                    s_true = val_loader.dataset.true_survival(t) 

                    pe = ((s_est - s_true).abs()/s_true.abs() * 100).mean()
                    l2 = ((s_est - s_true)**2).mean()
                    kl = ((s_true / s_est + 1e-9).log())**2
                    if use_kl:
                        l2 = kl
                    mse_matrix[idx_t, -1] = l2.item()

                else:
                    t = thresholds[idx_t] * torch.ones((1,d))
                    CDF = (-1 / t).exp()
                    s_true, s_est, l2, pe = cpm.survival_prob(CDF, val_loader.dataset)

                run_errs['nn'].append((l2.item(), pe.item(), s_est.item())) # nn errors

                for idx, estimator in enumerate(np_estimators):

                    np_survival = estimator.est_survival(CDF)

                    l2 = (s_true - np_survival)** 2
                    kl = ((s_true / np_survival + 1e-9).log()) ** 2
                    pe = (s_true - np_survival).abs() / s_true.abs() * 100
                    if use_kl:
                        l2 = kl
                    run_errs[np_list[idx]].append((l2.item(), pe.item(), np_survival.item()))
                    mse_matrix[idx_t, idx] = l2.item()

            argsorted_mse = torch.argsort(mse_matrix, dim=1, descending=False)
            top = torch.zeros(1, len(np_list) + 1)

            for mse_idx in range(len(np_list)+ 1):
                s = (argsorted_mse[:, 0] == mse_idx).sum()
                top[0,mse_idx] = s.item()

        else:
            top = None
            if d < 256:
                n = 10000
            else:
                n = 5000
            w = rand_simplex(n, d)
            a_true = train_loader.dataset.pickands(w) 
            a_mle = torch.max(cpm.a(w), w.max(1, keepdim=True)[0]).squeeze(1)

            l2 = F.mse_loss(a_mle, a_true)
            pe = (((a_mle - a_true) / a_true).abs()*100).mean()

            run_errs['nn'].append((l2.item(), pe.item())) # nn errors
            for idx, estimator in enumerate(np_estimators):
                with torch.no_grad():

                    np_a = estimator(w)

                    l2 = ((a_true - np_a)** 2).mean()
                    pe = ((a_true - np_a).abs() / a_true.abs() * 100).mean()
                    run_errs[np_list[idx]].append((l2.item(), pe.item()))

        results[run] = run_errs
        mse = []

        for key in run_errs.keys():
            avg = torch.tensor(run_errs[key]).mean(0)
            std = torch.tensor(run_errs[key]).std(0)
            print('{} L2: {:e} ({:e}), %: {:e}'.format(key, avg[0].item(), std[0].item(), avg[1].item()))
            if addkey:
                mse.append((key, avg[0].item(), std[0].item()))
            else:
                mse.append((avg[0].item(), std[0].item()))
            

        if d == 2:
            x = torch.linspace(0, 1-1e-8)
            x_ = torch.stack((x, 1 - x), dim=1)

            plt.style.use('seaborn-darkgrid')
            plt.plot(cpm.a(x_).detach().cpu().numpy(), label='NN')
            for idx, estimator in enumerate(np_estimators):
                if np_list[idx] == 'naive':
                    plt.plot(estimator(x_).detach().cpu().numpy(), label='Pickands')
                else:
                    plt.plot(estimator(x_).detach().cpu().numpy(), label=np_list[idx].upper())

            if type(train_loader.dataset) == SL or type(train_loader.dataset) == ASL:
                plt.plot(train_loader.dataset.pickands(x_).detach().cpu().numpy(), label='Truth')

            plt.tight_layout()
            plt.legend()
            plt.savefig('pickands_comp.pdf')
            plt.close('all')

    if ret_estimators:
        return results, mse, cpm, top, np_estimators
    else:
        return results, mse, cpm, top

def get_table(a_results, idx=0):
    '''
    Returns the table in latex format for the particular experiment
    '''
    import pandas as pd

    data_dict = {k: np.concatenate([np.array(dl[0][k])[:,idx] for dl in a_results]) for k in a_results[0][0]}
    df = pd.DataFrame(data_dict)
    df2 = df.mean().map(str) + " (" + df.std().map(str) + ") "

    LaTeX = ''

    print(df2)

def plot_results(a_mse, file_name, xlabel, title, results, xtick=None, log=True):
    '''
    Plots the results from the experiment
    '''
    plt.style.use('seaborn-darkgrid')
    for c in range(len(a_mse[0][1])):
        if type(a_mse[0][0] == torch.Tensor):
            x = np.array([i[0] for i in a_mse])
        else:
            x = np.array([i[0] for i in a_mse])
        # index 1 is {(mean, std)}
        mean =  np.array([i[1][c][0] for i in a_mse])
        mean[np.isnan(mean)] = 0 
        std  =  np.array([i[1][c][1] for i in a_mse])
        std[np.isnan(std)] = 0 
        label = list(results[0].keys())[c].upper()
        if label == 'NAIVE':
            label = 'Pickands'
        elif label == 'NN':
            label = 'Proposed'
        if c== 0:
            ls = '--'
        elif c == 1:
            ls = ':'
        elif c == 2:
            ls = '-'
        elif c == 3:
            ls = '-.'

        eb1 = plt.errorbar(x, (mean), yerr=std, label=label, ls=ls, capsize=5.0, linewidth=4.0, elinewidth=3.0, markeredgewidth=4.0)
        eb1[-1][0].set_alpha(0.4)

    if log:
        plt.yscale('log')
    plt.xlabel(xlabel, fontsize=15)
    if 'asl' not in file_name:
        plt.ylabel(r'MSE', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if xtick is not None:
        plt.xticks([i[0] for i in a_mse], xtick)
    plt.legend(prop={'size':15})
    plt.tight_layout()
    plt.savefig(file_name)
    plt.close('all')

def run_sl_iid_experiment_multi():

    d = 2
    n_runs = 50
    model_params = {
            'lr' : LR,
            'survival' : True,
            'use_swa' : False
            }

    # Data parameters
    data_params = {
            'd' : d,
            'n_samp' : 100,
            'const_marginals' : True
            }

    # Architecture parameters
    arch_params = {
            'input_size' : d,
            'depth' :DEPTH, #4,  # 4
            'width' :WIDTH, #16,#16 
            'cond_size' : 0, 
            'cond_width': 0,
            }


    a_mse = []
    data_type = 'sl'
    percentile = 0.75
    #percentile = 0.01
    thresholds = torch.linspace(-1/np.log(percentile), 100)

    for a in torch.linspace(0.01, 0.95, 10):
        mses=[]

        for run in range(n_runs):

            print('alpha = ', a)
            tl, vl = get_data(data_type, alpha=a, **data_params)
            results, mse, cpm, top = train(tl, vl, arch_params, model_params, trainer_params, thresholds, addkey=False)
            #a_mse.append((a.cpu().numpy(), mse))
            mses.append(np.array([[m[0] for m in mse]]))

            #a_mse.append((a.cpu().numpy(), mse))

        mse = np.concatenate(mses)
        a_mse.append((a.cpu().numpy(), np.stack((mse.mean(0), mse.std(0)),1)))

    with open('results/survival_sl_w={}d={}.pkl'.format(WIDTH, DEPTH),'wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'results': results},f)
    title = r'$A = A_{SL}$'
    xlabel = r'$\alpha$'
    file_name = 'results/survival_sl_{}_w={}d={}.pdf'.format(n_runs, WIDTH, DEPTH)
    plot_results(a_mse, file_name, xlabel, title, results)


def run_sl_iid_experiment_change_d():

    a_mse = []
    n_runs = 50
    dims = [256, 512, 784, 1024] 
    for ind, d in enumerate(dims):
        mses = []
        for _ in range(n_runs):

            model_params = {
                    'lr' : LR,
                    'survival' : False,
                    'use_swa' : False
                    }

            # Data parameters
            data_params = {
                    'd' : d,
                    'n_samp' : 100,
                    'const_marginals' : True
                    }

            # Architecture parameters
            arch_params = {
                    'input_size' : d,
                    'depth' : DEPTH,  # 3
                    'width' : WIDTH,# 24
                    'cond_size' : 0, 
                    'cond_width': 0,
                    }


            data_type = 'sl'
            thresholds = None

            a = 0.5

            print('alpha = ', a)
            tl, vl = get_data(data_type, alpha=a, **data_params)
            results, mse, cpm, _ = train(tl, vl, arch_params, model_params, trainer_params, thresholds, addkey=False)
            mses.append(np.array([[m[0] for m in mse]]).astype(np.float))


        mse = np.concatenate(mses)
        if len(dims) == 4:
            if d > 200 and d < 2040:
                plt.style.use('seaborn-darkgrid')
                #plt.yscale('log')
                if len(dims) > 4:
                    plt.subplot(2,2,ind-2)
                else:
                    plt.subplot(2,2,ind+1)
                df = pd.DataFrame(mse, columns=['CFG','Pickands','BDV','Proposed'])
                g = sns.boxplot(data=df, palette='pastel')
                g.set(title=r'$d=${}'.format(d))
                g.tick_params(bottom=False)
                plt.tight_layout()
                plt.savefig('results/box_sl_w={}d={}_Dd.pdf'.format(WIDTH, DEPTH))
            else:
                plt.close('all')
        a_mse.append((d, np.stack((mse.mean(0), mse.std(0)),1)))
        print(a_mse)

    plt.close('all')
    with open('results/mse_sl_d={}N={}E={}w={}d={}_Dd.pkl'.format(d, n_runs, trainer_params['max_epochs'], WIDTH, DEPTH),'wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'results': results},f)
    title = r'$A = A_{SL}$'
    xlabel = r'$d$'
    file_name = 'results/mse_sl_d={}N={}E={}w={}d={}_Dd.pdf'.format(d, n_runs, trainer_params['max_epochs'], WIDTH, DEPTH)
    plot_results(a_mse, file_name, xlabel, title, results,log=False)

def run_asl_iid_experiment_change_d():

    d = 256
    a_mse = []
    a = 0.5*torch.ones(1)
    n_runs = 50
    dims = [256, 512, 784, 1024] 
    for ind, d in enumerate(dims):
        mses = []

        for _ in range(n_runs):

            model_params = {
                    'lr' : LR,
                    'survival' : False,
                    'use_swa' : False
                    }

            # Data parameters
            data_params = {
                    'd' : d,
                    'n_samp' : 100,
                    'const_marginals' : True
                    }

            # Architecture parameters
            arch_params = {
                    'input_size' : d,
                    'depth' : DEPTH,  # 2
                    'width' : WIDTH,# 256 
                    'cond_size' : 0, 
                    'cond_width': 0,
                    }


            data_type = 'asl'
            thresholds = None

            print('alpha = ', a)
            tl, vl = get_data(data_type, alpha=a, **data_params)
            results, mse, cpm, _ = train(tl, vl, arch_params, model_params, trainer_params, thresholds, addkey=False)
            mses.append(np.array([[m[0] for m in mse]]))


        mse = np.concatenate(mses)
        if len(dims) == 4:
            if d > 200 and d < 2040:
                plt.style.use('seaborn-darkgrid')
                if len(dims) > 4:
                    plt.subplot(2,2,ind-2)
                else:
                    plt.subplot(2,2,ind+1)
                df = pd.DataFrame(mse, columns=['CFG','Pickands','BDV','Proposed'])
                g = sns.boxplot(data=df, palette='pastel')
                g.set(title=r'$d=${}'.format(d))
                g.tick_params(bottom=False)
                plt.tight_layout()
                plt.savefig('results/box_asl_w={}d={}_Dd.pdf'.format(WIDTH, DEPTH))
            else:
                plt.close('all')
        a_mse.append((d, np.stack((mse.mean(0), mse.std(0)),1)))

    plt.close('all')
    with open('results/mse_asl_d={}N={}E={}w={}d={}_Dd.pkl'.format(d, n_runs, trainer_params['max_epochs'], WIDTH, DEPTH),'wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'results': results},f)
    title = r'$A = A_{ASL}$'
    xlabel = r'$d$'
    file_name = 'results/mse_asl_d={}N={}E={}w={}d={}_Dd.pdf'.format(d, n_runs, trainer_params['max_epochs'], WIDTH, DEPTH)
    plot_results(a_mse, file_name, xlabel, title, results,log=False)

def run_sl_iid_experiment_a_change():

    a_mse = []
    n_runs = 50
    d = 256
    for ind, a in enumerate([0.25, 0.5, 0.75, 1]):
        mses = []
        for _ in range(n_runs):

            model_params = {
                    'lr' : LR,
                    'survival' : False,
                    'use_swa'  : False
                    }

            # Data parameters
            data_params = {
                    'd' : d,
                    'n_samp' : 100,
                    'const_marginals' : True
                    }

            # Architecture parameters
            arch_params = {
                    'input_size' : d,
                    'depth' : DEPTH,  # 3
                    'width' : WIDTH,# 24
                    'cond_size' : 0, 
                    'cond_width': 0,
                    }

            data_type = 'sl'
            thresholds = None

            print('alpha = ', a)
            tl, vl = get_data(data_type, alpha=a, **data_params)
            results, mse, cpm, _ = train(tl, vl, arch_params, model_params, trainer_params, thresholds, addkey=False)
            mses.append(np.array([[m[0] for m in mse]]))


        mse = np.concatenate(mses)
        plt.style.use('seaborn-darkgrid')
        plt.subplot(2,2,ind+1)
        if ind == 0:
            plt.ylim(0, 2.5e-4)
        df = pd.DataFrame(mse, columns=['CFG','Pickands','BDV','Proposed'])
        g = sns.boxplot(data=df, palette='pastel')
        g.set(title=r'$\alpha=${}'.format(a))
        g.tick_params(bottom=False)
        plt.tight_layout()
        plt.savefig('results/box_sl_w={}d={}_Da.pdf'.format(WIDTH,DEPTH))

        if a == 1:
            plt.close('all')
        a_mse.append((a, np.stack((mse.mean(0), mse.std(0)),1)))

    plt.close('all')
    with open('results/mse_sl_d={}N={}E={}w={}d={}_Da.pkl'.format(d, n_runs, trainer_params['max_epochs'], WIDTH, DEPTH),'wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'results': results},f)
    title = r'$A = A_{SL}$'
    xlabel = r'$d$'
    file_name = 'results/mse_sl_d={}N={}E={}w={}d={}_Da.pdf'.format(d, n_runs, trainer_params['max_epochs'], WIDTH, DEPTH)
    plot_results(a_mse, file_name, xlabel, title, results)


def run_sl_asl_samples():

    seed_everything(777)

    a_mse = []
    n_runs = 1
    d = 2
    #for ind, a_ in enumerate([0.25, 0.5, 0.75, 1]):
    for ind, a_ in enumerate([ 0.5, 0.75, 1]):
        for data_type in ['asl', 'sl']:
            a = a_*torch.ones(1)
            mses = []
            for _ in range(n_runs):

                model_params = {
                        'lr' : LR,
                        'survival' : False,
                        'use_swa'  : False
                        }

                # Data parameters
                data_params = {
                        'd' : d,
                        'n_samp' : 2000,
                        'const_marginals' : True
                        }

                # Architecture parameters
                arch_params = {
                        'input_size' : d,
                        'depth' : DEPTH,  # 3
                        'width' : WIDTH,# 24
                        'cond_size' : 0, 
                        'cond_width': 0,
                        }


                thresholds = None

                print('alpha = ', a)
                tl, vl = get_data(data_type, alpha=a, **data_params)
                results, mse, cpm, _, est = train(tl, vl, arch_params, model_params, trainer_params, thresholds, addkey=False, ret_estimators=True)
                samps = sample(cpm.net)
                samps_ = samps / samps.sum(-1, keepdims=True)
                real_samps = tl.dataset[:100][0].cpu()
                real_samps_ = real_samps/real_samps.sum(-1,keepdims=True)
                plt.scatter(samps_[:,0], samps_[:,1], alpha=0.25)
                plt.scatter(real_samps_[:,0], real_samps_[:,1], alpha=0.25)
                plt.savefig('results/{}_sample_{}.pdf'.format(data_type,a_))
                plt.close('all')
                a_true = tl.dataset.pickands
                for e in est:
                    get_all_bi_extremal_coeffs(e, d, 'results/{}_{}_extremal_{}.pdf'.format(data_type,a_, type(e).__name__), a_true=a_true)
                get_all_bi_extremal_coeffs(cpm.a, d, 'results/{}_{}_extremal_{}w={}d={}.pdf'.format(data_type,a_, type(cpm.net).__name__,WIDTH,DEPTH), a_true=a_true)


def run_asl_3d_plot():

    seed_everything(777)

    a_mse = []
    n_runs = 1
    d = 3
    for ind, a_ in enumerate([0.25, 0.5, 0.75, 1]):
        a = a_*torch.ones(1)
        mses = []
        for _ in range(n_runs):

            model_params = {
                    'lr' : LR,
                    'survival' : False,
                    'use_swa'  : False
                    }

            # Data parameters
            data_params = {
                    'd' : d,
                    'n_samp' : 10,
                    'const_marginals' : True
                    }

            # Architecture parameters
            arch_params = {
                    'input_size' : d,
                    'depth' : DEPTH,  # 3
                    'width' : WIDTH,# 24
                    'cond_size' : 0, 
                    'cond_width': 0,
                    }


            data_type = 'asl'
            thresholds = None

            print('alpha = ', a)
            tl, vl = get_data(data_type, alpha=a, **data_params)
            results, mse, cpm, _, est = train(tl, vl, arch_params, model_params, trainer_params, thresholds, addkey=False, ret_estimators=True)
            a_true = tl.dataset.pickands
            contour_3d(cpm.a, 'results/{}_{}_3d_{}.pdf'.format(data_type, a_, type(cpm.net).__name__))
            for e in est:
                contour_3d(e, 'results/{}_{}_3d_{}.pdf'.format(data_type, a_, type(e).__name__))
            contour_3d(a_true, 'results/{}_{}_3d_gt.pdf'.format(data_type, a_))

def run_asl_iid_experiment_a_change():

    a_mse = []
    n_runs = 50
    d = 256
    for ind, a_ in enumerate([0.25, 0.5, 0.75, 1]):
        a = a_*torch.ones(1)
        mses = []
        for _ in range(n_runs):

            model_params = {
                    'lr' : LR,
                    'survival' : False,
                    'use_swa'  : False
                    }

            # Data parameters
            data_params = {
                    'd' : d,
                    'n_samp' : 100,
                    'const_marginals' : True
                    }

            # Architecture parameters
            arch_params = {
                    'input_size' : d,
                    'depth' : DEPTH,  # 3
                    'width' : WIDTH,# 24
                    'cond_size' : 0, 
                    'cond_width': 0,
                    }


            data_type = 'asl'
            thresholds = None

            #for a in torch.linspace(0.01, 0.95, 20):
            #a = 0.5

            print('alpha = ', a)
            tl, vl = get_data(data_type, alpha=a, **data_params)
            results, mse, cpm, _ = train(tl, vl, arch_params, model_params, trainer_params, thresholds, addkey=False)
            mses.append(np.array([[m[0] for m in mse]]))
            #a_mse.append((a.cpu().numpy(), mse))

        mse = (np.concatenate(mses))
        plt.style.use('seaborn-darkgrid')
        #plt.yscale('log')
        plt.subplot(2,2,ind+1)
        df = pd.DataFrame(mse, columns=['CFG','Pickands','BDV','Proposed'])
        g = sns.boxplot(data=df, palette='pastel')
        g.set(title=r'$\alpha=${}'.format(a_))
        g.tick_params(bottom=False)
        plt.tight_layout()
        plt.savefig('results/box_asl_w={}d={}_Da.pdf'.format(WIDTH,DEPTH))

        if a == 1:
            plt.close('all')

        a_mse.append((a, np.stack((mse.mean(0), mse.std(0)),1)))
        print(a_mse)

    plt.close('all')
    with open('results/mse_asl_d={}N={}E={}w={}d={}_Da.pkl'.format(d, n_runs, trainer_params['max_epochs'], WIDTH, DEPTH),'wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'results': results},f)
    title = r'$A = A_{ASL}$'
    xlabel = r'$d$'
    file_name = 'results/mse_asl_d={}N={}E={}w={}d={}_Da.pdf'.format(d, n_runs, trainer_params['max_epochs'], WIDTH, DEPTH)
    plot_results(a_mse, file_name, xlabel, title, results)

def run_asl_iid_experiment_multi():

    d = 2
    n_runs = 50
    model_params = {
            'lr' : LR,
            'survival' : True,
            'use_swa' : False
            }

    # Data parameters
    data_params = {
            'd' : d,
            'n_samp' : 100,
            'const_marginals' : True
            }

    # Architecture parameters
    arch_params = {
            'input_size' : d,
            'depth' : DEPTH, #4,  # 4
            'width' : WIDTH, #16,#16 
            'cond_size' : 0, 
            'cond_width': 0,
            }


    a_mse = []
    data_type = 'asl'
    percentile = 0.75
    thresholds = torch.linspace(-1/np.log(percentile), 100)

    for a in torch.linspace(0.01, 0.95, 10):
        mses=[]

        for run in range(n_runs):

            print('alpha = ', a)
            tl, vl = get_data(data_type, alpha=a, **data_params)
            results, mse, cpm, top = train(tl, vl, arch_params, model_params, trainer_params, thresholds, addkey=False)
            mses.append(np.array([[m[0] for m in mse]]))

        mse = np.concatenate(mses)
        a_mse.append((a.cpu().numpy(), np.stack((mse.mean(0), mse.std(0)),1)))

    with open('results/survival_asl_w={}d={}.pkl'.format(WIDTH, DEPTH),'wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'results': results},f)
    title = r'$A = A_{ASL}$'
    xlabel = r'$\alpha$'
    file_name = 'results/survival_asl_{}_w={}d={}.pdf'.format(n_runs, WIDTH, DEPTH)
    plot_results(a_mse, file_name, xlabel, title, results)


def run_ozone_experiment():

    # Model Parameters
    csv_files = [
            'data/ozone_data/sequoia_kaweah_1984.csv', 
            'data/ozone_data/sequoia_ash_1984.csv',
            'data/ozone_data/sequoia_grove_1984.csv',
            'data/ozone_data/sequoia_lookout_1984.csv'
            ]
    locations = [
            'SEKI-LK_O3_PPB', 
            'SEKI-AS_O3_PPB',
            'SEKI-GG_O3_PPB',
            'SEKI-XX_O3_PPB'
            ]

    d = len(csv_files)
    model_params = {
            'lr' : LR,
            'survival' : True
            }

    # Data parameters
    data_params = {
            'd' : d,
            'n_samp' : 100,
            'const_marginals' : True
            }

    # Architecture parameters
    arch_params = {
            'input_size' : d,
            'depth' : DEPTH, 
            'width' : WIDTH, 
            'cond_size' : 0, 
            'cond_width': 0,
            }


    a_mse = []
    a_top = []
    a_results = []
    threshold = 0.5

    cvm_stat = {}
    month_pairs = [ (6,7), (7,8), (8,9) ]  
    loc_name = locations

    for idx, month_pair in enumerate(month_pairs):

        dataset = Ozone(csv_files, locations, month=month_pair[0], year=0, scale='D')
        val_dataset = Ozone(csv_files, locations, month=month_pair[1], year=0, scale='W')
        thresholds = val_dataset.get_thresholds(threshold)
        print('Num points train: ', len(dataset))
        print('Num points val:   ', len(val_dataset))
        tl = DataLoader(dataset, batch_size=1000, shuffle=False)
        vl = DataLoader(val_dataset, batch_size=1000, shuffle=False) 
        results, mse, cpm, top, est = train(tl, vl, arch_params, model_params, trainer_params, thresholds, ret_estimators=True)
        a_mse.append((idx, mse))
        a_results.append(results)
        a_top.append(top)

        get_all_bi_extremal_coeffs(cpm.a, d, 'ozone_extremal_{}_w{}d{}.pdf'.format(type(cpm.net).__name__, WIDTH, DEPTH), labels=loc_name, d_use=d)
        for e in est:
            get_all_bi_extremal_coeffs(e, d, 'ozone_extremal_{}.pdf'.format(type(e).__name__), labels=loc_name, d_use=d)

        for m in mse:
            key = m[0]
            try:
                cvm_stat[key].append(m[1])
            except KeyError:
                cvm_stat[key] = []
                cvm_stat[key].append(m[1])

    for key in cvm_stat.keys():
        means = np.array(cvm_stat[key])
        print(key)
        print('{:e} ({:e})'.format(means.mean(), means.std()))

    get_table(a_results)

    a_top = torch.cat(a_top)

    print(a_top)
    print(a_top.float().mean(0))
    print(a_top.float().std(0))

    with open('results/survival_ozone4.pkl','wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'results': results},f)
    title = r'MSE of survival probabilities'
    xlabel = 'Month pair'
    file_name = 'survival_ozone4.pdf'

def run_commodities_experiment():

    csv_files = [
            'data/new_commodities/Coffee.csv', 
            'data/new_commodities/Copper.csv', 
            'data/new_commodities/Corn.csv', 
            'data/new_commodities/Crude_Oil.csv', 
            'data/new_commodities/Gold.csv', 
            'data/new_commodities/Heating_Oil.csv', 
            'data/new_commodities/Natural_Gas.csv', 
            'data/new_commodities/Platinum.csv', 
            'data/new_commodities/Silver.csv', 
            'data/new_commodities/Wheat.csv', 
            ]

    c_c_g = [0,3,4]

    d = len(csv_files)
    model_params = {
            'lr' : LR,
            'survival' : True,
            'use_swa'  : False
            }

    # Data parameters
    data_params = {
            'd' : d,
            'n_samp' : 100,
            'const_marginals' : True
            }

    # Architecture parameters
    arch_params = {
            'input_size' : d,
            'depth' : DEPTH, 
            'width' : WIDTH, 
            'cond_size' : 0, 
            'cond_width': 0,
            }


    year_pairs = [ ([2015], [ 2016, 2017, 2018]), ([2016], [2017, 2018, 2019]), ([2017], [2018, 2019, 2020]) ]

    year_tick = [p[1][0] for p in year_pairs]

    a_mse = []
    a_results = []
    a_top = []

    cvm_stat = {}
    quantile = 0.5

    loc_name = [f.split('/')[-1][:-4] for f in csv_files]

    for idx, year_pair in enumerate(year_pairs):

        dataset = Commodities(csv_files, time_range=year_pair[0], scale='W', extreme_type='mdd')
        val_dataset = Commodities(csv_files, time_range=year_pair[1], scale='M', extreme_type='mdd')
        thresholds = val_dataset.get_thresholds(quantile)

        print('Num points train: ', len(dataset))
        print('Num points val:   ', len(val_dataset))
        tl = DataLoader(dataset, batch_size=1000, shuffle=False)
        vl = DataLoader(val_dataset, batch_size=1000, shuffle=False) 
        results, mse, cpm, top, est= train(tl, vl, arch_params, model_params, trainer_params, thresholds, ret_estimators=True)

        a_mse.append((idx, mse))
        a_results.append(results) # list of list of dicts, each entry in the dict is 100 length list with (l2, pe, est)
        a_top.append(top)

        contour_3d(cpm.a, 'results/commodities_3d_{}_w{}d{}.pdf'.format(type(cpm.net).__name__, WIDTH, DEPTH), d,c_c_g)
        for e in est:
            contour_3d(e, 'results/commodities_3d_{}.pdf'.format(type(e).__name__), d,c_c_g)

        get_all_bi_extremal_coeffs(cpm.a, d, 'comm_extremal_{}_w{}d{}.pdf'.format(type(cpm.net).__name__, WIDTH, DEPTH), labels=loc_name)
        for e in est:
            get_all_bi_extremal_coeffs(e, d, 'comm_extremal_{}.pdf'.format(type(e).__name__), labels=loc_name)

        for m in mse:
            key = m[0]
            try:
                cvm_stat[key].append(m[1])
            except KeyError:
                cvm_stat[key] = []
                cvm_stat[key].append(m[1])

    for key in cvm_stat.keys():
        means = np.array(cvm_stat[key])
        print(key)
        print('{:e} ({:e})'.format(means.mean(), means.std()))

    get_table(a_results)
    a_top = torch.cat(a_top)

    print(a_top)
    print(a_top.float().mean(0))
    print(a_top.float().std(0))

    with open('results/survival_commodities2.pkl','wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'results': results}, f)

    title = r'MSE of survival probabilities'
    xlabel = 'Year pair'
    file_name = 'survival_commodities2.pdf'

def run_sp_experiment():

    import os
    path = 'data/sp_data'
    csv_files = os.listdir(path)

    csv_files = [os.path.join(path,c) for c in csv_files]

    model_params = {
            'lr'       : LR,
            'survival' : True, 
            'use_swa'  : False
            }


    year_pairs = [ ([2015], [ 2016, 2017, 2018]), ([2016], [2017, 2018, 2019]), ([2017], [2018, 2019, 2020]) ]

    year_tick = [p[1][0] for p in year_pairs]

    quantile = 0.5

    a_mse = []
    a_results = []
    a_top = []
    cvm_stat = {}

    for idx, year_pair in enumerate(year_pairs):

        if idx == 0:
            t = True
        else: 
            t = False

        dataset = SP(csv_files, time_range=year_pair[0], train=t, scale='W', extreme_type='mdd')
        d = dataset[0][0].shape[0]
        val_dataset = SP(csv_files, time_range=year_pair[1], train=not t, scale='M', extreme_type='mdd')
        thresholds = val_dataset.get_thresholds(quantile, 1000)
        # Data parameters
        data_params = {
                'd'               : d,
                'n_samp'          : 100,
                'const_marginals' : True
                }

        # Architecture parameters
        arch_params = {
                'input_size' : d,
                'depth'      : DEPTH, 
                'width'      : WIDTH, 
                'cond_size'  : 0, 
                'cond_width' : 0,
                }

        print('Num points train: ', len(dataset))
        print('Num points val:   ', len(val_dataset))
        tl = DataLoader(dataset,     batch_size=1000, shuffle=False)
        vl = DataLoader(val_dataset, batch_size=1000, shuffle=False) 
        results, mse, cpm, top, est = train(tl, vl, arch_params, model_params, trainer_params, thresholds, ret_estimators=True)
        coin_type = ['{}'.format((os.path.basename(f).split('_')[0]).upper())  for f in csv_files]


        get_all_bi_extremal_coeffs(cpm.a, d, 'spy_extremal_{}_w{}d{}.pdf'.format(type(cpm.net).__name__, WIDTH,DEPTH), labels=coin_type, d_use=8)
        for e in est:
            get_all_bi_extremal_coeffs(e, d, 'spy_extremal_{}.pdf'.format(type(e).__name__), labels=coin_type, d_use=8)

        a_mse.append((idx, mse))
        a_results.append(results)
        a_top.append(top)

        for m in mse:
            key = m[0]
            try:
                cvm_stat[key].append(m[1])
            except KeyError:
                cvm_stat[key] = []
                cvm_stat[key].append(m[1])

    for key in cvm_stat.keys():
        means = np.array(cvm_stat[key])
        print(key)
        print('{:e} ({:e})'.format(means.mean(), means.std()))

    get_table(a_results)

    a_top = torch.cat(a_top)

    print(a_top)
    print(a_top.float().mean(0))
    print(a_top.float().std(0))

    with open('results/survival_sp.pkl','wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'a_results': a_results},f)


def run_crypto_experiment():

    import os
    path = 'data/crypto'
    csv_files = os.listdir('data/crypto')

    csv_files = [os.path.join(path,c) for c in csv_files]

    d = len(csv_files)
    model_params = {
            'lr'       : LR,
            'survival' : True, 
            'use_swa'  : False,
            }



    year_pairs = [ ([2015], [ 2016, 2017, 2018]), ([2016], [2017, 2018, 2019]), ([2017], [2018, 2019, 2020]) ]
    year_tick = [p[1][0] for p in year_pairs]

    quantile = 0.5

    a_mse = []
    a_results = []
    a_top = []
    cvm_stat = {}

    for idx, year_pair in enumerate(year_pairs):

        dataset     = Crypto(csv_files, time_range=year_pair[0], scale='W', extreme_type='mdd')
        val_dataset = Crypto(csv_files, time_range=year_pair[1], scale='M', extreme_type='mdd')
        thresholds = val_dataset.get_thresholds(quantile)
        d = dataset[0][0].shape[0]

        # Data parameters
        data_params = {
                'd'               : d,
                'n_samp'          : 100,
                'const_marginals' : True
                }

        # Architecture parameters
        arch_params = {
                'input_size' : d,
                'depth'      : DEPTH, 
                'width'      : WIDTH, 
                'cond_size'  : 0, 
                'cond_width' : 0,
                }

        print('Num points train: ', len(dataset))
        print('Num points val:   ', len(val_dataset))
        print('d = ', d)

        tl = DataLoader(dataset,     batch_size=1000, shuffle=False)
        vl = DataLoader(val_dataset, batch_size=1000, shuffle=False) 
        results, mse, cpm, top, est = train(tl, vl, arch_params, model_params, trainer_params, thresholds, ret_estimators=True)
        coin_type = ['{}'.format((os.path.basename(f).split('_')[0]).upper())  for f in csv_files]

        get_all_bi_extremal_coeffs(cpm.a, d, 'crypto_extremal_{}_w{}d{}.pdf'.format(type(cpm.net).__name__, WIDTH,DEPTH), labels=coin_type, d_use=8)
        for e in est:
            get_all_bi_extremal_coeffs(e, d, 'crypto_extremal_{}.pdf'.format(type(e).__name__), labels=coin_type, d_use=8)

        a_mse.append((idx, mse))
        a_results.append(results)

        a_top.append(top)

        for m in mse:
            key = m[0]
            try:
                cvm_stat[key].append(m[1])
            except KeyError:
                cvm_stat[key] = []
                cvm_stat[key].append(m[1])

    for key in cvm_stat.keys():
        means = np.array(cvm_stat[key])
        print(key)
        print('{:e} ({:e})'.format(means.mean(), means.std()))

    get_table(a_results)

    a_top = torch.cat(a_top)

    print(a_top)
    print(a_top.float().mean(0))
    print(a_top.float().std(0))
    with open('results/survival_crypto.pkl','wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'a_results': a_results},f)

    title = r'MSE of survival probabilities'
    xlabel = 'Year pair'
    file_name = 'survival_crypto.pdf'

def run_wind_experiment():

    import os
    path = 'data/wind_data'
    csv_files = os.listdir(path)

    csv_files = [os.path.join(path,c) for c in csv_files]

    d = len(csv_files)
    model_params = {
            'lr'       : LR,
            'survival' : True, 
            'use_swa'  : False,
            }

    # Data parameters
    data_params = {
            'd'               : d,
            'n_samp'          : 100,
            'const_marginals' : True
            }

    # Architecture parameters
    arch_params = {
            'input_size' : d,
            'depth'      : DEPTH,#3,  
            'width'      : WIDTH,#24, 
            'cond_size'  : 0, 
            'cond_width' : 0,
            }

    month_pairs = [ (6,7), (7,8), (8,9) ]  
    month_tick = [p[0] for p in month_pairs]
    month_tick = [ 'June', 'July', 'August' ]

    a_mse = []
    a_top = []
    a_results = []
    cvm_stat = {}

    quantile = 0.5

    loc_name = ['{}'.format((os.path.basename(f).split('_')[0]).upper())  for f in csv_files]
    for idx, month_pair in enumerate(month_pairs):

        print(month_pair)
        print('Quantile: {}'.format(quantile))
        dataset     = California(csv_files, month=month_pair[0], scale='D') 
        val_dataset = California(csv_files, month=month_pair[1], scale='W')
        thresholds = val_dataset.get_thresholds(quantile)

        print('Num points train: ', len(dataset))
        print('Num points val:   ', len(val_dataset))
        tl = DataLoader(dataset, batch_size=1000, shuffle=False)
        vl = DataLoader(val_dataset, batch_size=1000, shuffle=False) 
        results, mse, cpm, top, est = train(tl, vl, arch_params, model_params, trainer_params, thresholds, ret_estimators=True)

        a_top.append(top)
        a_mse.append((idx, mse))
        a_results.append(results)

        get_all_bi_extremal_coeffs(cpm.a, d, 'cali_extremal_{}_w{}d{}.pdf'.format(type(cpm.net).__name__,WIDTH,DEPTH), labels=loc_name)
        for e in est:
            get_all_bi_extremal_coeffs(e, d, 'cali_extremal_{}.pdf'.format(type(e).__name__), labels=loc_name)

        for m in mse:
            key = m[0]
            try:
                cvm_stat[key].append(m[1])
            except KeyError:
                cvm_stat[key] = []
                cvm_stat[key].append(m[1])

    for key in cvm_stat.keys():
        means = np.array(cvm_stat[key])
        print(key)
        print('{:e} ({:e})'.format(means.mean(), means.std()))

    get_table(a_results)

    a_top = torch.cat(a_top)

    print(a_top)
    print(a_top.float().mean(0))
    print(a_top.float().std(0))

    with open('results/survival_wind.pkl','wb') as f:
        import pickle
        pickle.dump({'a_mse': a_mse, 'results': a_results},f)

    title = r'MSE of survival probabilities'
    xlabel = 'Month pair'
    file_name = 'survival_wind.pdf'

def real_data_exp():

    seed_everything(777)

    # Real Experiments
    print('WIND')
    run_wind_experiment()
    print('+++++++++++++++++')
    print('OZONE')
    run_ozone_experiment()
    print('+++++++++++++++++')
    print('COMMODITIES')
    run_commodities_experiment()
    print('+++++++++++++++++')
    print('SPY')
    run_sp_experiment()
    print('+++++++++++++++++')
    print('CRYPTO')
    run_crypto_experiment()
    print('+++++++++++++++++')

# Real Experiments
real_data_exp()

# Synthetic Experiments
# (survival)
run_sl_iid_experiment_multi()
run_asl_iid_experiment_multi()

# (pickands mse)
run_sl_iid_experiment_change_d()
run_sl_iid_experiment_a_change()
run_asl_iid_experiment_change_d()
run_asl_iid_experiment_a_change()

# (plots)
#run_sl_asl_samples()
#run_asl_3d_plot()

