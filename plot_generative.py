import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import os
import pickle
import csv

import numpy as np

ex_type = 'da'

BASE_PATH = 'logs_generative'
data_types = ['sl', 'asl']
alphas = np.linspace(0,1,9)
dd = np.array([ 8, np.sqrt(128), 16, 28, 32 ]) **2 
ex_name = lambda d, alpha, data : "d={}_a={}_data={}".format(d, alpha, data)
d = 225
v = 2
if ex_type == 'dd':
    alpha = 0.5
    EX_NAMES = [ex_name(int(d), alpha, exp) for exp in data_types for d in dd]  # define the experiment paths
    iterator = dd
else:
    d = 225
    EX_NAMES = [ex_name(d, alpha, exp) for exp in data_types for alpha in alphas]  # define the experiment paths
    iterator = alphas

exp_data = {}
mses = np.zeros((len(EX_NAMES), 3))
stds = np.zeros((len(EX_NAMES), 3))

for idx, exp in enumerate(EX_NAMES):

    path = os.path.join(os.path.join(BASE_PATH, exp), 'version_{}/'.format(v))
    print(path)

    # first read the last line of the log file
    data = {}
    with open(path + 'metrics.csv', 'r') as f:
        last_row = list(csv.reader(f))[-1]
        data['mse_gen'] = float(last_row[3])
        data['std_gen'] = float(last_row[4])
        data['mse_real'] = float(last_row[6])
        data['std_real'] = float(last_row[7])

    # then read the dMNN data
    with open(path + 'dmnn.pkl', 'rb') as f:
        dmnn = pickle.load(f)
        data['mse_dmnn'] = dmnn['dmnn_mse']
        data['std_dmnn'] = dmnn['dmnn_std']

    exp_data[iterator[idx // len(data_types)]] = data.copy()

    mses[idx, 0] = data['mse_gen']
    mses[idx, 1] = data['mse_dmnn']
    mses[idx, 2] = data['mse_real']

    stds[idx, 0] = data['std_gen']
    stds[idx, 1] = data['std_dmnn']
    stds[idx, 2] = data['std_real']

    exp_idx = idx // len(iterator) 

    plt.style.use('seaborn-darkgrid')
    mean = mses[len(iterator)*exp_idx:idx+1, :]
    std  = stds[len(iterator)*exp_idx:idx+1, :]
    aas = iterator[:idx - len(iterator)*exp_idx + 1]
    plt.errorbar(aas, mean[:,0], yerr=std[:,0], label='Generator', linewidth=3, capthick=5, markeredgewidth=4, capsize=5)
    plt.errorbar(aas, mean[:,1], yerr=std[:,1], label='dMNN', linewidth=3, capthick=5, markeredgewidth=4, capsize=5)
    plt.errorbar(aas, mean[:,2], yerr=std[:,2], label='Real', linewidth=3, capthick=5, markeredgewidth=4, capsize=5)
    plt.yscale('log')

    if exp_idx == 0:
        plt.ylim(1e-8, 9e-1)
    else:
        plt.ylim(1e-6, 3e-1)
    if ex_type == 'dd':
        plt.xlabel(r'$d$', fontsize=15)
    else:
        plt.xlabel(r'$\alpha$', fontsize=15)
    plt.ylabel(r'$\| A - A_{CFG} \|$', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(prop={'size':15})
    plt.tight_layout()
    plt.savefig('{}_CFG_sampling_{}.pdf'.format(data_types[exp_idx], ex_type))
    plt.close('all')

