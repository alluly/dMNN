import pickle
import os

import torch
import torch.distributions as tdist
from torch.utils.data import Dataset, DataLoader

import statsmodels
from statsmodels.tsa.stattools import adfuller

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
np.random.seed(seed=777)

from scipy.stats import genextreme as gev
from scipy.stats import gumbel_r, weibull_max, weibull_min

import utils
from utils import rand_sl, rand_asl, rand_simplex

from datetime import timedelta

cuda = False
if torch.cuda.is_available() and cuda:
    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
else:
    torch.set_default_tensor_type('torch.DoubleTensor')

class SLProcess(Dataset):

    def __init__(self, d, n_samp, alpha=torch.linspace(0,1,100), const_marginals=False, valid=False):
        super(SLProcess, self).__init__()
        eps = 1e-6

        self.alpha = alpha
        self.n_samp= n_samp
        self.valid = valid

        self.data  = torch.stack([rand_sl(a, n_samp, d) for a in alpha]) # shape: time t, num samples at time t, dim

        if const_marginals:
            order  = torch.argsort(self.data, dim=0, descending=False)
            self.F = ((torch.argsort(order, dim=0).double() + 1) / (len(alpha)*n_samp + 1)).squeeze(0)
        else:
            assert n_samp > 1, 'Please use multiple runs'
            order  = torch.argsort(self.data, dim=1, descending=False)
            self.F = ((torch.argsort(order, dim=1).double() + 1) / (self.data.shape[1] + 1)).squeeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # make the idx the time step
        ts = torch.tensor(idx).repeat(self.n_samp).double() / self.data.shape[0]
        a  = self.alpha[idx].repeat(self.n_samp)

        if self.valid:
            return (self.data[idx], self.F[idx], ts, a)

        return (self.data[idx], self.F[idx], ts)


class Process(Dataset):
    def __init__(self, n_samp, data, pickands, const_marginals, valid=False):
        super(Process, self).__init__()

        eps = 1e-6

        self.pickands = pickands
        self.n_samp   = n_samp
        self.valid    = valid

        if const_marginals:
            order  = torch.argsort(self.data, dim=0, descending=False)
            self.F = ((torch.argsort(order, dim=0).double() + 1) / (self.data.shape[0]*n_samp + 1)).squeeze(0)
        else:
            assert n_samp > 1, 'Please use multiple runs'
            order  = torch.argsort(self.data, dim=1, descending=False)
            self.F = ((torch.argsort(order, dim=1).double() + 1) / (self.data.shape[1] + 1)).squeeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # make the idx the time step
        ts = torch.tensor(idx).repeat(self.n_samp).double() / self.data.shape[0]

        if self.valid:
            w  = rand_simplex(100, self.data.shape[2])
            if self.data.shape[2] == 2:
                x = torch.linspace(1e-8, 1-1e-8)
                w = torch.stack((x, 1 - x), dim=1)
            a  = self.pickands[idx](w)
            return (self.data[idx], self.F[idx], ts, w, a)

        return (self.data[idx], self.F[idx], ts)

    def true_survival(self, CDF, idx):
        '''
        u : threshold : 2D vector of thresholds shape = (1, 2)
        This function exactly calculates the survival prob on copula:
        P [F(X_1) > u1, F(X_2) > u2]
        The exact formula is P [F(X_1) > u1, F(X_2) > u2] = 1 - u1 - u2 + C(u1, u2)
        '''
        # compute marginal CDFs (Frechet)
        u = CDF

        # create symmetric logistic dis.
        w = u.log() / u.log().sum()
        C = (u.log().sum() * self.pickands[idx](w)).exp()
        survival = 1 - u.sum() + C
        return survival

class SLProcess(Process):

    def __init__(self, 
            d, 
            n_samp, 
            alpha=torch.linspace(0,1,100), 
            const_marginals=True, 
            valid=False):

        self.n_samp= n_samp
        sl = [rand_sl(a, d, n_samp) for a in alpha]

        self.data     = torch.stack([s[0] for s in sl]) # shape: time t, num samples at time t, dim
        self.pickands = [s[1] for s in sl] 

        super(SLProcess, self).__init__(n_samp, self.data, self.pickands, const_marginals, valid)

class ASLProcess(Process):

    def __init__(self, 
            n_samp, 
            theta, 
            alpha, 
            const_marginals=False, 
            valid=False):

        asl = [rand_asl(alpha[i,:], theta, n_samp) for i in range(alpha.shape[0])]

        self.data     = torch.stack([a[0] for a in asl]) # shape: time t, num samples at time t, dim
        self.pickands = [a[1] for a in asl] 

        super(ASLProcess, self).__init__(n_samp, self.data, self.pickands, const_marginals, valid)


class SL(Dataset):

    def __init__(self, d, n_samp=1, alpha=0.5):
        super(SL, self).__init__()
        eps = 1e-6

        self.alpha = alpha
        self.n_samp= n_samp
        self.data, self.pickands = rand_sl(alpha, d, n_samp) 

        order  = torch.argsort(self.data, dim=0, descending=False)
        self.F = ((torch.argsort(order, dim=0).double() + 1) / (n_samp + 1)).squeeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.F[idx])

    def true_survival(self, CDF):
        '''
        u : threshold : 2D vector of thresholds shape = (1, 2)
        This function exactly calculates the survival prob on SL copula:
        P [F(X_1) > u1, F(X_2) > u2]
        The exact formula is P [F(X_1) > u1, F(X_2) > u2] = 1 - u1 - u2 + C(u1, u2)
        '''
        # compute marginal CDFs (Frechet)
        u = CDF

        # create symmetric logistic dis.
        w = u.log() / u.log().sum()
        C = (u.log().sum() * self.pickands(w)).exp()
        survival = 1 - u.sum() + C
        return survival

class ASL(Dataset):

    def __init__(self, alphas, theta, n_samp=1):
        super(ASL, self).__init__()
        eps = 1e-6

        self.n_samp = n_samp
        self.alpha = alphas[0]
        self.data, self.pickands = rand_asl(alphas, theta, n_samp)

        order  = torch.argsort(self.data, dim=0, descending=False)
        self.F = ((torch.argsort(order, dim=0).double() + 1) / (n_samp + 1)).squeeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.F[idx])

    def true_survival(self, CDF):
        '''
        u : threshold : 2D vector of thresholds shape = (1, 2)
        This function exactly calculates the survival prob on SL copula:
        P [F(X_1) > u1, F(X_2) > u2]
        The exact formula is P [F(X_1) > u1, F(X_2) > u2] = 1 - u1 - u2 + C(u1, u2)
        '''
        u = CDF

        # create symmetric logistic dis.
        w = u.log() / u.log().sum()
        C = (u.log().sum() * self.pickands(w)).exp()
        survival = 1 - u.sum() + C
        return survival

class ScaledDirichlet(Dataset):
    def __init__(self, d, alpha, rho, n_samp=10000):
        assert alpha.shape[0] == d

        super(ScaledDirichlet, self).__init__()

        self.data = torch.zeros(n_samp, d)
        self.dir = tdist.dirichlet.Dirichlet(alpha)

        gamma = tdist.gamma.Gamma(alpha, 1)
        gamma_rho  = tdist.gamma.Gamma(alpha + rho, 1)
        exp = tdist.exponential.Exponential(1)

        c = torch.lgamma(alpha + rho).exp() / torch.lgamma(alpha).exp()

        d_set = set(range(d))

        for idx in range(n_samp):

            E = exp.sample()
            Y = torch.zeros(d)

            while 1 / E > Y.min():
                J = set([torch.randint(d, [1]).item()])
                Z = torch.zeros(d)
                #Z[list(J)] = gamma.sample()[list(J)] + gamma_rho.sample()
                Z[list(J)] = gamma_rho.sample()[list(J)] 
                Z[list(d_set - J)] = gamma.sample()[list(d_set - J)]

                W = Z ** rho / c
                S = W / W.sum()
                #Z = Z / Z.sum()
                #S = Z / c

                Y = torch.maximum(Y, d * S / E)
                E = E + exp.sample()

            #self.data[idx,:] = torch.exp( -1 / Y ).clone()
            self.data[idx,:] = Y.clone()

        order  = torch.argsort(self.data, dim=0, descending=False)
        self.F = ((torch.argsort(order, dim=0).double() + 1) / (n_samp + 1)).squeeze(0)

    def l(self, x):
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx], self.F[idx])

class CauchyFlight(Dataset):
    def __init__(self, d, n_samp=1):
        super(CauchyFlight, self).__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class EmpiricalDataset(Dataset):
    def __init__(self, data, use_gumbel=False, use_weibull=False, use_frechet=False):
        super(EmpiricalDataset, self).__init__()

        self.data = data
        self.use_gumbel = use_gumbel


        # fit margins
        self.F = np.zeros_like(self.data, dtype=float)

        # n margins x [shape, loc, scale]
        self.margin_params = np.zeros((self.data.shape[1], 3))

        for loc in range(self.data.shape[1]):

            # get GEV params 
            m = self.data[:,loc].mean()
            g_init = gumbel_r.fit(self.data[:, loc]) 
            c = gev.fit(self.data[:, loc], loc=g_init[0], scale=g_init[1])

            # get CDF transform of params
            if use_gumbel:
                self.F[:, loc]  = gumbel_r.cdf(self.data[:, loc], *g_init)
                self.margin_params[loc, :] = np.array([0., g_init[0], g_init[1]])
            elif use_weibull:
                w_init = weibull_max.fit(self.data[:, loc]) 
                self.F[:, loc]  = weibull_max.cdf(self.data[:, loc], *w_init)
                self.margin_params[loc, :] = np.array([ w_init[0], w_init[1], w_init[2]])
            elif use_frechet:
                f_init = weibull_min.fit(self.data[:, loc]) 
                self.F[:, loc]  = weibull_min.cdf(self.data[:, loc], *f_init)
                self.margin_params[loc, :] = np.array([ f_init[0], f_init[1], f_init[2]])
            else:
                self.F[:, loc]  = gev.cdf(self.data[:, loc], *c)
                self.margin_params[loc, :] = np.array([c[0], c[1], c[2]])
            
            # plot marginals
            #plt.hist(self.data[:,loc], density=True)
            #x = np.linspace(weibull_min.ppf(0.01, *f_init), weibull_min.ppf(0.99, *f_init), 100)
            #plt.plot(x, weibull_min.pdf(x, *f_init))
            #plt.savefig('marginal_{}_est.pdf'.format(loc))
            #plt.close('all')

        self.F = torch.Tensor(self.F)
        self.data = torch.Tensor(self.data)
        self.sorted, _ = torch.sort(self.data, dim=0, descending=False)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return (self.data[idx], self.F[idx])

    def true_survival(self, threshold=120):
        # return the number of events above the threshold
        rate = (self.data >= threshold).float().prod(1).mean()

        #rate = (self.data >= threshold).float().mean()
        #rate = num / self.data.shape[0]
        return rate

    def get_thresholds(self, quantile, n_samp=1000):
        '''
        Returns the data thresholds for a specific quantile
        quantile = 1 returns the max and quantile = 0 returns the min
        '''
        u = self.sorted[0,:].cpu()
        normed = self.data.norm(2, dim=1)
        idx_max = torch.argmax(normed)
        l = self.data[idx_max, :].cpu()

        threshold = np.linspace(l, u, n_samp)

        return threshold

class California(EmpiricalDataset):
    def __init__(self, csv, month=0, year=0, col:str = 'WindGust', remove_neg: bool = True, scale: str = None):

        col_names = ['Date', 
                'Year', 
                'DoY', 
                'DoR', 
                'SRTotal', 
                'WindAvg',
                'VectorDeg', 
                'WindGust', 
                'AvgTemp', 
                'MaxTemp', 
                'MinTemp', 
                'AvgHum', 
                'MaxHum', 
                'MinHum', 
                'Precip']

        df0 = 0

        for idx, file in enumerate(csv):
            df = pd.read_csv(file, sep='\s+', parse_dates=True, skipfooter=4, names=col_names)
            df = df[['Date', col]]

            if idx == 0:
                df0 = df
            else:
                df0 = df0.merge(df, on='Date')

        df0['Date'] = pd.to_datetime(df['Date'])
        df0.set_index('Date', inplace=True)

        # taking maxima over a day (change 'D' by 'M' if you need monthly maxima)
        if month: 
            freq = 'D'
        elif year:
            freq = 'M'
        else:
            freq = 'Y'

        if scale is None:
            scale = freq

        grouped = df0.groupby(pd.Grouper(freq=freq))
        grouped_max = grouped.max()

        # constrain data to month
        if month: 
            grouped_max = grouped_max.loc[(grouped_max.index.month == month)]

        if year:
            grouped_max = grouped_max.loc[(grouped_max.index.year == year)]

        # remove rows with negative values
        if remove_neg:
            grouped_max = grouped_max[grouped_max > 0]

        grouped_max.dropna(inplace=True)
        data = grouped_max.values

        super(California, self).__init__(data)

class Ozone(EmpiricalDataset):
    def __init__(self, csv_files: str, locations: str, month: int = 6, year: int = 0, remove_neg: bool = True, scale: str = None):
        '''
        csv_files is a list of csv locations
        csv_files contains 20 years of data Jan 2000- Dec 2019
        to avoid time dealing with time dependency we can restrict
        the data to 10 years for e.g.
        The threshold for ozone is taken to be t = 120 ppb as denoted
        by the EPA as an unhealthy level.
        '''

        assert month * year == 0, 'choose either month or year to group by'
        data_dict = {loc:[] for loc in locations}
        data_array = [] 

        df0 = 0

        for idx, file in enumerate(csv_files):
            # read csv file 
            df = pd.read_csv(file, delimiter=",", parse_dates=True)
            df = df.drop(columns=['Unnamed: 2'])

            if idx == 0:
                df0 = df
            else:
                df0 = df0.merge(df, on='DATE_TIME')


        df0['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
        df0.set_index('DATE_TIME', inplace=True)

        # taking maxima over a day (change 'D' by 'M' if you need monthly maxima)
        if month: 
            freq = 'D'
        elif year:
            freq = 'M'
        else:
            freq = 'Y'

        if scale is None:
            scale = freq

        grouped = df0.groupby(pd.Grouper(freq=scale))
        grouped_max = grouped.max()

        # constrain data to month
        if month: 
            grouped_max = grouped_max.loc[(grouped_max.index.month == month)]

        if year:
            grouped_max = grouped_max.loc[(grouped_max.index.year == year)]

        # remove rows with negative values
        if remove_neg:
            grouped_max = grouped_max[grouped_max > 0]

        grouped_max.dropna(inplace=True)
        data = grouped_max.values

        super(Ozone, self).__init__(data)

        # empirical estimates (not needed)
        #order  = torch.argsort(self.data, dim=0, descending=False)
        #self.F = ((torch.argsort(order, dim=0).double() + 1) / (self.data.shape[0] + 1)).squeeze(0)


class IB(EmpiricalDataset):

    def __init__(self, csv, train: bool = True, scale: str = 'H'):

        df0 = 0
        for idx, c in enumerate(csv):
            df = pd.read_csv(c, parse_dates=True)
            df = df[['Open']]

            df['log_ret'] = np.log(df.Open) - np.log(df.Open.shift(1))
            df = df.drop(columns=['Open'])
            if idx == 0:
                df0 = df
            else:
                df0 = df0.merge(df, left_index=True, right_index=True)

        grouped = df0.groupby(pd.Grouper(freq=scale))
        grouped_max = grouped.max()

        grouped_max.dropna(inplace=True)
        n = grouped_max.values.shape[0]
        
        if train:
            data = grouped_max.values[:n // 2]
        else:
            data = grouped_max.values[n // 2:]

        super(IB, self).__init__(data, use_gumbel=True)

class Commodities(EmpiricalDataset):

    def __init__(self, csv_files:str, time_range, extreme_type: str = 'log', scale: str = 'M'):

        df0 = 0

        for idx, file in enumerate(csv_files):
            # read the cdv file

            if extreme_type == 'log':
                df = pd.read_csv(file, delimiter=",", parse_dates=True, thousands=',')
                df['log_ret'] = np.log(df.Price) - np.log(df.Price.shift(1))
                df = df.drop(columns=['Change %', 'Price', 'Open', 'High', 'Low', 'Vol.'])
            elif extreme_type == 'mdd':
                df = pd.read_csv(file, delimiter=",", parse_dates=True, thousands=',')

                df['Price'] = df['Price'].astype(float)

                df = df.drop(columns=['Change %', 'Open', 'High', 'Low', 'Vol.'])

            else:
                df = pd.read_csv(file, delimiter=",", parse_dates=True)

                # extract % change values
                for i in range(len(df)):
                    df.iloc[i]['Change %'] = df.iloc[i]['Change %'][:-1]

                # take the negative of values
                df['Change %'] = df['Change %'].astype(float)

                # keep the % Change column only
                df = df.drop(columns=['Price', 'Open', 'High', 'Low', 'Vol.'])

            # select time range
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df = df[df.index.year.isin(time_range)]

            # get maxima
            grouped = df.groupby(pd.Grouper(freq=scale))
            if extreme_type == 'mdd':
                grouped_max = (grouped.min() - grouped.max())/grouped.max()
            else:
                grouped_max = grouped.max()

            # merge dataframe
            if idx == 0:
                df0 = grouped_max
            else:
                df0 = df0.merge(grouped_max, on='Date')

        # dropna
        df0 = df0.dropna()

        data = df0.values

        for i in range(data.shape[1]):
            test_stationarity(data[:,i])

        super(Commodities, self).__init__(data)

class SP(EmpiricalDataset):

    def __init__(self, csv_files, time_range, train: bool = True, extreme_type: str = 'log', scale: str = 'M'):
        df0 = 0

        for idx, file in enumerate(csv_files):
            # read the cdv file
            df = pd.read_csv(file, delimiter=",", parse_dates=True)

            # alternatively calc log returns
            col_name = '{}'.format((os.path.basename(file).split('.')[0]).upper())

            if extreme_type == 'log':
                # calculate log returns
                df[col_name] = np.log(df.close) - np.log(df.close.shift(1))
            elif extreme_type == 'mdd':
                # calculate max drawdown
                df[col_name] = (df.close) 
            else:
                # calculate percent change 
                df[col_name] = df.close.pct_change()
                df[col_name] = df[col_name].astype(float)

            # keep the % Change column only
            dc = ['adjusted_close', 'open','high','low','close','volume','dividend_amount','split_coefficient']
            df = df.drop(columns=dc)

            # select time range
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            if ((df['timestamp'].max() - df['timestamp'].min() ) < timedelta(days=5000)):
                continue

            df.set_index('timestamp', inplace=True)
            df = df[df.index.year.isin(time_range)]

            # get maxima
            grouped = df.groupby(pd.Grouper(freq=scale))
            if extreme_type == 'log' or extreme_type == 'pct':
                grouped_max = grouped.max()
            elif extreme_type == 'mdd':
                grouped_max = -((grouped.min() - grouped.max()) / grouped.max())

            # merge dataframe
            if type(df0) == int:
                df0 = grouped_max
            else:
                df0 = df0.merge(grouped_max, on='timestamp')

        f=open('sp_names.txt','w')
        for items in list(df0):
            f.writelines(items + '\n') 
        f.close()

        # dropna
        df0 = df0.dropna()

        data = df0.values
        n_samp = data.shape[0]


        #if train: 
        #    data = data[:n_samp//2]
        #else:
        #    data = data[n_samp//2:]

        '''
        df0.plot()
        plt.title('Crypto Log Returns Monthly Max')
        plt.ylabel(r'$\log ( p_{t+1} / p_t ) $')
        plt.legend()
        plt.savefig('crypto.pdf')
        plt.close('all')
        '''

        #for i in range(data.shape[1]):
        #    test_stationarity(data[:,i])

        super(SP, self).__init__(data)

class Crypto(EmpiricalDataset):
    def __init__(self, csv_files, time_range, extreme_type: str = 'log', scale: str = 'D'):
        df0 = 0

        for idx, file in enumerate(csv_files):
            # read the cdv file
            df = pd.read_csv(file, delimiter=",", parse_dates=True)

            # alternatively calc log returns
            col_name = '{}'.format((os.path.basename(file).split('_')[0]).upper())

            if extreme_type == 'log':
                # calculate log returns
                df[col_name] = np.log(df.Close) - np.log(df.Close.shift(1))
            elif extreme_type == 'mdd':
                # calculate max drawdown
                df[col_name] = (df.Close) 
            else:
                # calculate percent change 
                df[col_name] = df.Close.pct_change()
                df[col_name] = df[col_name].astype(float)

            # keep the % Change column only
            df = df.drop(columns=['Open', 'Close', 'High', 'Low', 'Volume', 'Market Cap'])

            # select time range
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            if df.shape[0] < 2000:
                continue
            df = df[df.index.year.isin(time_range)]

            # get maxima
            grouped     = df.groupby(pd.Grouper(freq=scale))
            if extreme_type == 'log' or extreme_type == 'pct':
                grouped_max = grouped.max()
            elif extreme_type == 'mdd':
                grouped_max = (-((grouped.min() - grouped.max()) / grouped.max())).abs()

            # merge dataframe
            if type(df0) == int:
                df0 = grouped_max
            else:
                df0 = df0.merge(grouped_max, on='Date')

        f=open('crypto_names.txt','w')
        for items in list(df0):
            f.writelines(items + '\n') 

        f.close()

        # dropna
        df0 = df0.dropna()

        data = df0.values

        '''
        df0.plot()
        plt.title('Crypto Log Returns Monthly Max')
        plt.ylabel(r'$\log ( p_{t+1} / p_t ) $')
        plt.legend()
        plt.savefig('crypto.pdf')
        plt.close('all')
        '''

        #for i in range(data.shape[1]):
        #    test_stationarity(data[:,i])
        super(Crypto, self).__init__(data)

class CryptoMin(EmpiricalDataset):
    def __init__(self, csv_files, time_range, use_log_ret: bool = True, scale: str = 'H'):
        df0 = 0

        for idx, file in enumerate(csv_files):
            # read the cdv file
            df = pd.read_csv(file, delimiter=",", parse_dates=True)

            # alternatively calc log returns
            col_name = '{}'.format((os.path.basename(file).split('_')[0]).upper())

            if use_log_ret:
                df[col_name] = np.log(df.close) - np.log(df.close.shift(1))

            else:
                # take the negative of values
                df[col_name] = df.close.pct_change()
                df[col_name] = df[col_name].astype(float)

            df = df[1:]

            # keep the % Change column only
            df = df.drop(columns=[k for k in df.keys() if k != col_name and k != 'time'])
            #df = df.drop(columns=['open', 'close', 'high', 'low', 'volume'])

            # select time range
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            df.set_index('time', inplace=True)
            #df = df[df.index.year.isin(time_range)]

            if len(df) < 10000:
                continue

            # get maxima
            grouped = df.groupby(pd.Grouper(freq=scale, origin='start_day'))
            grouped_max = grouped.max()

            if grouped_max.isnull().sum().sum() > 5000:
                continue

            if type(df0) == int:
                df0 = grouped_max
            else:
                df0 = df0.merge(grouped_max, on='time')

        # dropna
        df0 = df0.dropna()

        data = df0.values

        '''
        df0.plot()
        plt.title('Crypto Log Returns Monthly Max')
        plt.ylabel(r'$\log ( p_{t+1} / p_t ) $')
        plt.legend()
        plt.savefig('crypto.pdf')
        plt.close('all')
        '''

        #for i in range(data.shape[1]):
        #    test_stationarity(data[:,i])

        super(CryptoMin, self).__init__(data, use_gumbel=True)

def test_stationarity(data):
    adf_test = adfuller(data, autolag='AIC')
    print(adf_test)

if __name__=='__main__':
    import timeit
    import seaborn as sns
    start_time = timeit.default_timer()
    sd = ScaledDirichlet(d=3, alpha=torch.tensor([1/5,1/5,1/5]), rho=1/5, n_samp=10000)
    data = (sd.data[:] / sd.data[:].sum(-1,keepdims=True)).cpu()
    import pandas as pd
    data = pd.DataFrame(data[:,:2].numpy(),columns=['x','y'])
    sns.kdeplot(data=data, x='x', y='y', shade = True, cmap = "PuBu")
    plt.savefig('sd.pdf')
    print('Exact sim time: {}'.format(timeit.default_timer() - start_time))
    exit()

    csv_files = ['data/ozone_data/sequoia_kaweah.csv', 'data/ozone_data/sequoia_ash.csv']
    locations = ['SEKI-LK_O3_PPB', 'SEKI-AS_O3_PPB']
    o = Ozone(csv_files, locations, month=0)
    print(o.get_thresholds(0.15))
    exit()

    cfg = utils.CFGEstimator(samples=o.data)
    x = torch.linspace(0, 1-1e-8)
    x_ = torch.stack((x, 1 - x), dim=1)
    p = cfg(x_)
    plt.plot(p.cpu())
    plt.savefig('cfg_ozone.pdf')
    exit()

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt

    data = SLProcess(2, 1, alpha=(torch.sin(torch.linspace(1,2,500))))
    #plt.plot(data[:][0][:,:,1]- data[:][0][:,:,0],'b')
    plt.yscale('log')
    plt.plot(data[1:][0])#,'b')
    data2 = SLProcess(2, alpha=torch.zeros(500)+0.5)
    #plt.plot(data2[:][0][:,:,1] - data2[:][0][:,:,0],'r')
    plt.plot(data2[:][0][:,0])#,'r')
    plt.savefig('sl_process.pdf')

