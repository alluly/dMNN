import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
from numpy import euler_gamma
from scipy.special import expi, gammainc

from datetime import datetime
def rand_exp(*dims):
    return -torch.rand(*dims).log()

def rand_simplex(batch_size, dim):
    exp = rand_exp(batch_size, dim)
    return exp / torch.sum(exp, dim=1, keepdim=True)


def rand_positive_stable(alpha, *dims):
    U = math.pi*torch.rand(*dims)
    W = rand_exp(*dims)

    return (torch.sin(alpha * U) / (U.sin() ** (1 / alpha))) * (torch.sin((1-alpha)*U) / W) ** (1/alpha - 1)

def rand_sym_log(n_samples, dim, alpha):
    if alpha > 0:
        S = rand_positive_stable(alpha, n_samples, 1)
        W = rand_exp(n_samples, dim)
        return (S / W) ** alpha
    else:
        return torch.ones(1, dim) / rand_exp(n_samples, 1)

class ExpIntegralEi(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.as_tensor(numpy_expi(x.detach().numpy()))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output * torch.exp(x) / x

expi = ExpIntegralEi.apply

class LogIntegral(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return -torch.as_tensor(numpy_expi(x.detach().log().numpy()))

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return grad_output / torch.log(x)

logintegral = LogIntegral.apply



class AsymmetricLogisticCopula():
    def __init__(self, alphas, thetas):
        self.m = alphas.shape[0]
        assert thetas.shape[0] == self.m, \
            'Number of alphas {} different from number of thetas {}'.format(self.m, thetas.shape[0])
        self.dim = thetas.shape[1]
        assert torch.all(thetas >= 0)
        if torch.any(thetas.sum(dim=0) != 1.):
            warn("thetas columns do not sum to 1, rescaling")
            thetas /= thetas.sum(dim=0, keepdim=True)
        self.alphas = alphas.view(1, -1, 1)
        self.thetas = thetas.unsqueeze(0)

    def sample(self, n_samples):
        Sm = rand_positive_stable(self.alphas, n_samples, self.m, 1)
        Wm = rand_exp(n_samples, self.m, self.dim)
        Xm = self.thetas * torch.where(self.alphas > 0, (Sm / Wm) ** self.alphas,
                                       torch.ones(1, 1, self.dim) / rand_exp(n_samples, self.m, 1))
        return Xm.max(dim=1)[0]

    def pickand(self, w):
        wtheta = w.unsqueeze(1) * self.thetas
        out_alpha_pos = torch.sum(wtheta ** (1. / self.alphas), dim=2, keepdim=True) ** self.alphas
        out_alpha_zero = torch.max(wtheta, dim=2, keepdim=True)[0]
        return torch.sum(torch.where(self.alphas > 0, out_alpha_pos, out_alpha_zero), dim=1).squeeze()


class SymmetricLogisticCopula():
    def __init__(self, dim, alpha):
        self.dim = dim
        self.alpha = alpha

    def sample(self, n_samples):
        if self.alpha > 0:
            S = rand_positive_stable(self.alpha, n_samples, 1)
            W = rand_exp(n_samples, self.dim)
            return (S / W) ** self.alpha
        else:
            return torch.ones(1, self.dim) / rand_exp(n_samples, 1)

    def pickand(self, w):
        if self.alpha > 0:
            return torch.sum(w ** (1 / self.alpha), dim=1) ** self.alpha
        else:
            return torch.max(w, dim=1)[0]

def rand_asl(alphas, thetas, batch_size):

    asl = AsymmetricLogisticCopula(alphas, thetas)

    return asl.sample(batch_size), asl.pickand

def rand_sl(alpha, dim, batch_size):

    sl = SymmetricLogisticCopula(dim, alpha)

    return sl.sample(batch_size), sl.pickand

class MetaCE(nn.Module):
    def __init__(self, samples, est_F=None, survival=False):
        super(MetaCE, self).__init__()
        (self.n_samples, self.dim) = samples.shape
        self.samples = samples.T.unsqueeze(0)
        order = torch.argsort(self.samples, dim=2, descending=False)
        F_ = torch.argsort(order, dim=2).double()
        if est_F is None:
            est_F = 'n+1'
        try:
            est_F = float(est_F)
        except ValueError:
            if est_F == 'n+1':
                self.F = (F_ + 1) / (self.n_samples + 1)
            else:
                self.F = est_F.T
        else:
            assert 0 <= est_F <= 1
            self.F = (F_ + est_F) / self.n_samples
        self.survival = survival
        if self.survival: 
            self.F = 1 - self.F

    def est_survival(self, CDF):
        '''
        estimator : a subclass of metace
        theshold : d dimensional vector of thresholds
        Given an estimator compute survival prob from threshold
        '''
        assert self.survival, 'must be survival copula'

        t = (1-CDF).log()
        w = t / t.sum()
        A = self(w)
        survival = (t.sum()*A).exp()
        return survival

class MadogramEstimator_old(MetaCE):
    def __init__(self, samples, est_F=None):
        super(MadogramEstimator_old, self).__init__(samples, est_F)

    def forward(self, w):
        Fw = self.F ** (1/w.unsqueeze(2))
        v = torch.mean(Fw.max(dim=1, keepdim=True)[0] - Fw.mean(dim=1, keepdim=True), dim=2, keepdim=True).squeeze()
        c = torch.mean(w / (1 + w), dim=1)
        A = (v + c) / (1 - v - c)
        return A

class MadogramEstimator(MetaCE):
    def __init__(self, samples, est_F=None, survival=False):
        super(MadogramEstimator, self).__init__(samples, est_F, survival)
        self.logF = self.F.log()

    def forward(self, w):
        Fw = (self.logF /w.unsqueeze(2)).exp_()
        v = torch.mean(Fw.max(dim=1, keepdim=True)[0] - Fw.mean(dim=1, keepdim=True), dim=2, keepdim=True).squeeze()
        c = torch.mean(w / (1 + w), dim=1)
        A = (v + c) / (1 - v - c)
        return A

class NaiveEstimator(MetaCE):
    def __init__(self, samples, est_F=None, survival=False):
        super(NaiveEstimator, self).__init__(samples, est_F, survival)
        self.logF = self.F.log()

    def forward(self, w):
        xi = (-self.logF/w.unsqueeze(2)).min(dim=1)[0]
        hA = (-xi.log().mean(dim=1) - euler_gamma).exp()
        return hA

class CFGEstimator(MetaCE):
    def __init__(self, samples, lambda_fun=None, est_F=None, survival=False):
        super(CFGEstimator, self).__init__(samples, est_F, survival)
        self.logF = (self.F + 1e-6).log()
        xi_ek = (-self.logF / torch.eye(self.dim).unsqueeze(2)).min(dim=1)[0]
        self.loghA_ek = -(xi_ek + 1e-6).log().mean(dim=1)
        if lambda_fun is None:
            lambda_fun = lambda x: x
        self.lambda_fun = lambda_fun

    def forward(self, w):
        xi = (-self.logF/w.unsqueeze(2)).min(dim=1)[0] + 1e-6
        hA = (-xi.log().mean(dim=1) - self.lambda_fun(w) @ self.loghA_ek).exp()
        return hA

class CopulaEstimator(MetaCE):
    def __init__(self, samples, est_F=None, eps=1e-8, mode='add'):
        super(CopulaEstimator, self).__init__(samples, est_F)
        self.eps = torch.tensor(eps)
        self.mode = mode

    def forward(self, w):
        out = torch.all(self.F<w.unsqueeze(2), dim=1).double().mean(dim=1)
        if self.mode=='add':
            out = out + self.eps
        elif self.mode=='max':
            out = out.max(self.eps)
        return out

class BDVEstimator(MetaCE):
    def __init__(self, samples, g=None, est_F=None, survival=False):
        super(BDVEstimator, self).__init__(samples, est_F, survival)
        if g is None:
            g = lambda x: x
        self.g = g
        self.logF = self.F.log()
        range = torch.arange(self.n_samples, dtype=torch.get_default_dtype())
        self.log_mult = torch.log1p(1./range)
        self.log_mult[0] = 0

    def forward(self, w):
        xi_lu = (self.logF / w.unsqueeze(2)).max(dim=1)[0]
        xi = xi_lu.sort(dim=1)[0].exp()
        return self.g(xi) @ self.log_mult

class BDVEstimator1(BDVEstimator):
    def __init__(self, samples, est_F=None, k=0):
        super(BDVEstimator1, self).__init__(samples, est_F=est_F)
        self.g = lambda x: (k+1) * x ** (k+1)

class BDVEstimator2(BDVEstimator):
    def __init__(self, samples, est_F=None, k=0):
        super(BDVEstimator2, self).__init__(samples, est_F=est_F)
        def g(x):
            out_numpy = -expi((k+1)*x.detach().log().numpy())
            return torch.as_tensor(out_numpy) * (k+1)
        self.g = g

class BDVEstimatorMM(MetaCE):
    def __init__(self, samples, g=None, h=None, est_F=None, survival=False):
        super(BDVEstimatorMM, self).__init__(samples, est_F, survival=survival)
        if g is None or h is None:
            g = lambda x: x
            h = lambda x: x * (1 - torch.log(x))
        self.g = g
        self.h = h
        self.logF = self.F.detach().log()
        range = torch.arange(1, self.n_samples, dtype=torch.get_default_dtype())
        self.steps = range.unsqueeze(0) / self.n_samples
        self.log_steps = -torch.log(self.steps).squeeze()

    def forward(self, w):
        xi_lu = (self.logF / w.detach().unsqueeze(2)).max(dim=1)[0]
        xi = xi_lu.sort(dim=1)[0].exp()
        maxw = torch.max(w, dim=1)[0]
        out = self.h(xi[:,0]) + (1 - self.h(xi[:,-1])) * maxw
        lower = torch.max(xi[:, :-1], torch.min(xi[:, 1:], self.steps ** (1./maxw.unsqueeze(1))))
        out += maxw * torch.sum(self.h(lower) - self.h(xi[:,:-1]), dim=1)
        upper = torch.min(xi[:, 1:], torch.max(xi[:, :-1], self.steps))
        out += torch.sum(self.h(xi[:, 1:]) - self.h(upper), dim=1)
        out += (self.g(upper) - self.g(lower)) @ self.log_steps
        return out

class BDVEstimatorMM1(BDVEstimatorMM):
    def __init__(self, samples, est_F=None, k=0):
        super(BDVEstimatorMM1, self).__init__(samples, est_F=est_F)
        self.g = lambda x: (k+1) * x ** (k+1)
        self.h = lambda x: (1 - (1 + k) * x.log()) * x ** (k+1)

class BDVEstimatorMM2(BDVEstimatorMM):
    def __init__(self, samples, est_F=None, k=0, survival=False):
        super(BDVEstimatorMM2, self).__init__(samples, est_F=est_F, survival=survival)
        def g(x):
            out_numpy = -expi((k+1)*x.detach().log().numpy())
            return torch.as_tensor(out_numpy) * (k+1)
        self.g = g
        self.h = lambda x: x ** (k+1)

class BDVEstimatorLM(BDVEstimator):
    def __init__(self, samples, g=None, est_F=None):
        super(BDVEstimatorLM, self).__init__(samples, g, est_F)
        self.f_ek = 1 - super(BDVEstimatorLM, self).forward(torch.eye(self.dim))

    def forward(self, w):
        f_w = super(BDVEstimatorLM, self).forward(w)
        return f_w + w @ self.f_ek

def est_survival(estimator, threshold):
    '''
    estimator : a subclass of metace
    theshold : d dimensional vector of thresholds
    Given an estimator compute survival prob from threshold
    '''
    CDF = (-1 / threshold).exp()
    t = (1-CDF).log()
    w = t / t.sum()
    A = estimator(w)
    survival = (t.sum()*A).exp()

    return survival

def init_estimators(estimators, dataset, survival=False, est_F=None):
    est = []

    if 'cfg' in estimators:
        start = datetime.now()
        cfg = CFGEstimator(dataset, survival=survival, est_F=est_F)
        print('CFG time : {}'.format(datetime.now() - start))
        est.append(cfg)

    if 'naive' in estimators:
        naive = NaiveEstimator(dataset, survival=survival, est_F=est_F)
        est.append(naive)

    if 'bdv' in estimators:
        bdv = BDVEstimatorMM(dataset, survival=survival, est_F=est_F)
        est.append(bdv)

    return est

class GEV(torch.distributions.distribution.Distribution):
    def __init__(self, loc, scale, shape):
        super(GEV, self).__init__()
        self.loc   = loc
        self.scale = scale
        self.shape = shape


