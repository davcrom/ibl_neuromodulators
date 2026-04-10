# %% imports
import numpy as np
from numpy import linalg

from scipy import sparse
import scipy.stats as stats
from tqdm import tqdm

from RRRlib import *
from plotters import *

import matplotlib.pyplot as plt

def add_noise(A, dist):
    """ dist is (frozen) callable from scipy.stats.distributions """
    A += dist.rvs(size=A.shape)
    return A

def inject_cov(A):
    """ injects covariance into matrix 
    TODO write down precisely how and why
    TODO test me
    """

    data_rvs = stats.distributions.uniform(loc=0, scale=1).rvs
    density = cov_params['density']
    C = sparse.random(n_regressors, n_regressors, density=density, data_rvs=data_rvs).toarray() # was 0.1
    np.fill_diagonal(C, 1)
    C = np.tril(C) + np.triu(C.T, 1)
    X = X @ C
    
    return A

def dense_data_gen(n_features, n_samples, n_regressors, true_rank, dist):
    """ data """
    X = dist(n_samples, n_regressors)
    L = dist(n_regressors, true_rank)
    W = dist(true_rank, n_features)
    return X, L, W

def sparse_data_gen(n_features, n_samples, n_regressors, true_rank, sparse_params=None):
    """
    random data - for RRR benchmarking
    """

    # n_features = 20 # 'm' = number of neurons
    # n_samples = 1000 # 'n' = number of time samples
    # n_regressors = 20 # 'l' = number of all lags x events
    # true_rank = 10 # 'r'

    data_rvs = stats.distributions.norm(loc=loc,scale=scale).rvs

    loc = sparse_params['loc'] # was 1
    scale = sparse_params['scale'] # was 1
    density = sparse_params['density'] # was 0.3
    X = sparse.random(n_samples, n_regressors, density=density, data_rvs=data_rvs).toarray()
    L = sparse.random(n_regressors, true_rank,  density=density, data_rvs=data_rvs).toarray()
    W = sparse.random(true_rank, n_features,  density=density, data_rvs=data_rvs).toarray()

    B = L @ W
    Y = X @ B

    return Y, X, B

def rand_data_gen(n_features, n_samples, n_regressors, true_rank, noise_sig=0, is_sparse=False, cov=False, sparse_params=None, cov_params=None):
    """
    random data - for RRR benchmarking
    """

    # data generation flags
    # sparse = True
    # cov = True

    # n_features = 20 # 'm' = number of neurons
    # n_samples = 1000 # 'n' = number of time samples
    # n_regressors = 20 # 'l' = number of all lags x events
    # true_rank = 10 # 'r'
    # noise_sig = 2.0

    if is_sparse is False:
        X = np.random.randn(n_samples, n_regressors)
        L = np.random.randn(n_regressors, true_rank)
        W = np.random.randn(true_rank, n_features)
    else:
        # sparsity attempt
        loc = sparse_params['loc'] # was 1
        scale = sparse_params['scale'] # was 1
        density = sparse_params['density'] # was 0.3
        data_rvs = stats.distributions.norm(loc=loc,scale=scale).rvs
        X = sparse.random(n_samples, n_regressors, density=density, data_rvs=data_rvs).toarray()
        L = sparse.random(n_regressors, true_rank,  density=density, data_rvs=data_rvs).toarray()
        W = sparse.random(true_rank, n_features,  density=density, data_rvs=data_rvs).toarray()

    if cov: # injecting covariance in regressors
        # C = np.random.rand(n_regressors, n_regressors)
        data_rvs = stats.distributions.uniform(loc=0, scale=1).rvs
        density = cov_params['density']
        C = sparse.random(n_regressors, n_regressors, density=density, data_rvs=data_rvs).toarray() # was 0.1
        np.fill_diagonal(C, 1)
        C = np.tril(C) + np.triu(C.T, 1)
        X = X @ C

    B = L @ W
    Y = X @ B

    Y = Y + np.random.randn(*Y.shape) * noise_sig
    return Y, X, B

# USAGE
# data_dict = dict(
#     n_features = 100, # 'm' = number of neurons
#     n_samples = 200, # 'n' = number of time samples
#     n_regressors = 50, # 'l' = number of all lags x events
#     true_rank = 20, # 'r'
#     noise_sig = 5.0
#     )

# sparse_dict = dict(loc=1,
#                    scale=2,
#                    density=0.1)

# cov_dict = dict(density=0.2)

# Y, X, B = data_gen(**data_dict, is_sparse=True, sparse_params=sparse_dict, cov=True, cov_params=cov_dict)




# %%


def times2inds(tvec, times):
    return [np.argmin(np.absolute(tvec - t)) for t in times]

def binarize(tvec, times): # TODO refactor my name
    bvec = np.zeros(tvec.shape[0])
    bvec[times2inds(tvec, times)] = 1
    return bvec

def poisson_process(rate, t_min=0, t_max=1):
    dist = stats.distributions.expon(loc=0, scale=rate)
    n_samples = int((t_max / rate)*2) # should be sufficient
    tstamps = np.cumsum(dist.rvs(size=n_samples))
    binds = np.logical_and(tstamps > t_min, tstamps < t_max)
    return tstamps[binds]

# def kernel_gen(mu, sig, dt):
#     kvec = np.arange(-sig*5, sig*5, dt)
#     K = stats.distributions.norm(mu, sig).pdf(kvec)
#     K = K / np.sum(K)
#     return K

# def kernel_gen(mu, sig, n=100):
#     kvec = np.linspace(-sig*5, sig*5, n)
#     K = stats.distributions.norm(mu, sig).pdf(kvec)
#     K = K / np.sum(K)
#     return K

def kernel_gen(kvec, mu, sig):
    K = stats.distributions.norm(mu, sig).pdf(kvec)
    K = K / np.sum(K)
    return K

# %% the possible cases

"""
one latent over time = ongoing computation, i.e. reward expectation
event-kernel can contribute to latent
"""

"""
kernel to event map
"""

# %% make L
n_samples = 5000
dt = 1 
tvec = np.arange(0, n_samples*dt, dt)
n_events = 5
n_kernels = n_events + 1 # more kernels than events
true_rank = n_kernels

rates = np.random.rand(n_kernels) / 5
true_times = []

for rate in rates:
    tstamps = poisson_process(1/rate, t_min=0, t_max=n_samples*dt)
    true_times.append(tstamps)

rates2events_map = dict(zip(range(n_events), range(n_events)))
rates2events_map[n_kernels-1] = 0

kvec = np.linspace(-49,50,100)
kernels = []
mus = np.random.randn(n_kernels) * 5
mus[0] = -10
mus[1] = +10
sigs = (np.random.rand(n_kernels)+1)*5
sigs[0] *= 2

for i in range(n_kernels):
    kernels.append(kernel_gen(kvec, mus[i], sigs[i]))

L = np.empty((n_samples, true_rank))
for i, j in rates2events_map.items():
# for i in range(n_events):
    # K = kernel_gen(mus[i], sigs[i], dt)
    L[:,i] = np.convolve(binarize(tvec, true_times[j]), kernels[i], mode='same')

# matshow(L.T)

# %% plot kernels
fig, axes = plt.subplots(ncols=n_events)
for i, j in rates2events_map.items():
    axes[j].plot(kernels[i])


# %% make W
n_features = 100

loc = 1
scale = 1
density = 0.3
dist = stats.distributions.norm(loc=loc,scale=scale)
W = sparse.random(true_rank, n_features,  density=density, data_rvs=dist.rvs).toarray()

# %% THE DATA GENERATION
Y = L @ W 
noise_sig = 0.10
noise_dist = stats.distributions.norm(loc=0, scale=noise_sig)
Y = add_noise(Y, noise_dist)

# there is more to this here!
# L = X @ Kp

# %%
# DATA GEN IS OVER HERE


# %% the model matrix setup
def lag_regressor(reg, n_lags):
    lags = np.linspace(-n_lags/2, n_lags/2 - 1,n_lags).astype('int32')
    rolls = []
    for lag in lags:
        rolls.append(np.roll(reg, lag+1)) # the +1 is from the LM(L,X) deduced
    reg_ex = np.stack(rolls).T
    return reg_ex

def lag_combine(tvec, tstampss, n_lags):
    lagregs = []
    for tstamps in tstampss:
        lagregs.append(lag_regressor(binarize(tvec, tstamps), n_lags))
    X = np.concatenate(lagregs, axis=1)
    return X

n_lags = 200
X = lag_combine(tvec, true_times[:n_events], n_lags)








# %% inferring Kp
lam = ridge_lambda_est_CV(LM, L, X, args=())
Kp = LM(L, X, lam=lam)
plot_stacked_lines(Kp, yscl=8)

# %%
Kp_true = np.zeros(Kp.shape)
for i in range(len(kernels)):
    print(i)
    # h = int(kernels[0].shape[0]/2)
    # kernels[i][int(h-n_lags/2):int(h+n_lags/2)] # this suffers from rounding ix errors
    start_ix = int(kernels[0].shape[0]/2 - n_lags/2)
    stop_ix = start_ix + n_lags
    Kp_true[n_lags*i:n_lags*(i+1), i] = kernels[i][start_ix:stop_ix]

axes = plot_stacked_lines(Kp, yscl=8)
plot_stacked_lines(Kp_true, yscl=8, axes=axes, color='r')

# %% rank estimation

# idea - PCA style first estimation
# brute +- 5 ranks around that
# idea - subsampling Y to reduce computation time and do it real
# cv style

# %% brute rank estimation
ranks = np.arange(3,8)
errs = np.empty((ranks.shape[0], 5))

for i, r in enumerate(tqdm(ranks)):

    lam = ridge_lambda_est_CV(LM_rr_direct, Y, X, args=(r,))
    B_hats, res = LM_CV(LM_rr_direct, Y, X, (lam, r), cv=5)

    # lam = ridge_lambda_est_CV(LM_rr_direct, Y, X, args=(r, ))
    # B_hat = LM_rr_direct(Y, X, lam, r)
    errs[i] = res  #Rss(Y, X@B_hat)

# %%
fig, axes = plt.subplots()
axes.plot(ranks, errs)

# %%
th = 0.99
A = B_hat
from sklearn.decomposition import PCA
""" estimate rank by explained variance on PCA """
pca = PCA(n_components=A.shape[1])
pca.fit(A)
var_exp = np.cumsum(pca.explained_variance_ratio_) < th
rank_est = 1 + np.sum(var_exp)
print(rank_est)

ths = np.linspace(0.1,1.0,100)
var_exps = [1+np.sum(np.cumsum(pca.explained_variance_ratio_) < th) for th in ths]
plt.plot(ths, var_exps)

# %% or set
r = true_rank

# %% the regression part
r = n_events + 1
lam = ridge_lambda_est_CV(LM_rr_direct, Y, X, args=(r, ), n_samples=100)
B_hat = LM_rr_direct(Y, X, lam, r)

# this doesn't to anything! (GOOD)
U, S, Vh = reduce_rank(*SVD(B_hat),r, return_matrix=True)
B_hat_lr = U @ S @ Vh

# %% simple ridge and potentially faster alternative?
lam = ridge_lambda_est_CV(LM, Y, X, args=(), n_samples=500)
B_hat = LM(Y, X, lam)
U, S, Vh = reduce_rank(*SVD(B_hat),r, return_matrix=True)
B_hat_lr = U @ S @ Vh


# %%
Y_hat = X @ B_hat
Y_hat_lr = X @ B_hat_lr

# %%
axes = plot_stacked_lines(Y, yscl=1)
plot_stacked_lines(Y_hat, yscl=1, axes=axes, color='r')
plot_stacked_lines(Y_hat_lr, yscl=1, axes=axes, color='c')


# %% L and W retrieval

# %% old
ks = []
ws = []
for i in range(n_events):
    P = B_hat_lr[n_lags*i:n_lags*(i+1),:]
    k = np.average(P,axis=1)[:,np.newaxis]
    k = k / k.max()
    w = linalg.pinv(k)@P
    
    ks.append(k)
    ws.append(w)

W_hat = np.concatenate(ws)

# %% kernel retrieval
ks = []
ws = []
stol = 0.5
r = n_events + 1

for i in range(r):
    # splitting B_hat into each set of lagged regressors
    P = B_hat_lr[n_lags*i:n_lags*(i+1),:]

    # find out if multiple sub kernels per event
    U, s, Vh = SVD(P, return_matrix=False)
    n_ = np.sum(s > stol) # number of "sub" kernels in P

    # for each, retrieve the weights
    for j in range(n_): # TODO refactor me
        # k = np.sum(U[:, :j+1],1)[:,np.newaxis]
        k = U[:, j][:,np.newaxis]

        
        # norm: peak = pos
        if k[np.argmax(np.absolute(k))] < 0:
            k = k * -1
        # k = k / k.max()

        w = linalg.pinv(k)@P

        ks.append(k)
        ws.append(w)

W_hat = np.concatenate(ws)

# %% kernel retrieval with orthogonal NMF
# https://github.com/salar96/MEP-Orthogonal-NMF

# %%
Ww, H, model = ONMF_DA.func(P, 1)
plt.plot(Ww)

# %%
import sys
sys.path.append("/home/georg/code/MEP-Orthogonal-NMF/")
from MEP_ONMF import ONMF_DA

ks = []
ws = []
stol = 0.2
r = n_events + 1

for i in range(n_events):
    # splitting B_hat into each set of lagged regressors
    P = B_hat_lr[n_lags*i:n_lags*(i+1),:]

    # find out if multiple sub kernels per event
    U, s, Vh = SVD(P, return_matrix=False)
    n_ = np.sum(s > stol) # number of "sub" kernels in P

    if n_ == 1:
        # W, H, model = ONMF_DA.func(P, 1)
        # k = W[:,0][:,np.newaxis]
        k = U[:,0][:,np.newaxis]
        # norm: peak = pos
        if k[np.argmax(np.absolute(k))] < 0:
            k = k * -1
        # k = k / k.max()

        w = linalg.pinv(k)@P

        ks.append(k)
        ws.append(w)
        continue
    else:
        print(i)
        # for each, retrieve the weights
        Ww, H, model = ONMF_DA.func(P, n_)
        for j in range(n_): # TODO refactor me
            k = Ww[:, j][:,np.newaxis]
            
            # norm: peak = pos
            if k[np.argmax(np.absolute(k))] < 0:
                k = k * -1
            # k = k / k.max()

            w = linalg.pinv(k)@P

            ks.append(k)
            ws.append(w)

W_hat = np.concatenate(ws)

# %%
K_hat = B_hat_lr @ linalg.pinv(W_hat)
fig, axes = plt.subplots()
axes.plot(K_hat)

# %%
W_hat = np.concatenate(ws)
matshow(W)
matcompare(W, W_hat)

# %%
# %%
Y_hatat = X @ K_hat @ W_hat 

axes = plot_stacked_lines(Y, yscl=1)
axes = plot_stacked_lines(Y_hat, yscl=1, axes=axes, color='r')
axes = plot_stacked_lines(Y_hat_lr, yscl=1, axes=axes, color='c')
axes = plot_stacked_lines(Y_hatat, yscl=1, axes=axes, color='b')


# %%
i = 0
P = B_hat_lr[n_lags*i:n_lags*(i+1),:]
matshow(P)
W, H, model = ONMF_DA.func(P, 2)
plt.figure()
plt.plot(W)

# U, s, Vh = SVD(P, return_matrix=False)



# %%


# matcompare(W @ H.T, P.T)
# %%



# %% calculating L from tt and Kp
Kp = LM(y,X,0)
plt.plot(Kp)
X@Kp

# %%
fig, axes = plt.subplots(ncols=2)
axes[0].matshow(Y)
axes[1].matshow(X@Kp@w_units)
[ax.set_aspect('auto') for ax in axes]


# %%
lagregs = []
for i in range(n_events):
    bvec = np.zeros(n_samples)
    bvec[tt[i]] = 1
    lagregs.append(lag_regressor(bvec, n_lags))

X = np.concatenate(lagregs,axis=1)

# %%
r = n_events
U, S, Vh = reduce_rank(*SVD(Y), r, return_matrix=True)
Y_lr = U @ S @ Vh

# %%
Q, R = linalg.qr(Y_lr)
# plt.matshow(R[:r,:])

plt.matshow(Q[:,:r])
plt.gca().set_aspect('auto')
plt.figure()
plt.matshow(y)
plt.gca().set_aspect('auto')







# B_true = 

# %%
lam = ridge_lambda_est_CV(LM, Y, X, args=())
B_hat = LM_rr_direct(Y, X, lam, 5)

fig, axes = plt.subplots()
axes.matshow(B_hat)
axes.set_aspect('auto')

# %%
yscl = 1.0
fig, axes = plt.subplots()
for i in range(Y.shape[1]):
    axes.plot(Y[:,i] * yscl + i, color='k', lw=1)

Y_hat = X @ B_hat

for i in range(Y.shape[1]):
    axes.plot(Y_hat[:,i] * yscl + i, color='r', lw=1)




# %%
# B_hat = LM_enet(Y, X, alpha=0.5, l1_ratio=0.01)
r = n_events
B_hat = LM_rr_direct(Y, X, lam, r)
U, S, Vh = reduce_rank(*SVD(B_hat),r, return_matrix=True)
B_hat_lr = U @ S @ Vh
fig, axes = plt.subplots(ncols=2)
axes[0].matshow(B_hat)
axes[1].matshow(B_hat_lr)
[ax.set_aspect('auto') for ax in axes]

# %%
fig, axes = plt.subplots()
axes.matshow(Vh)

# %% fishy kernel retrieval
from sklearn.cluster import KMeans
Km = KMeans(n_clusters=r)
Km.fit(Vh.T)

fig, axes = plt.subplots(r)
for i in range(r):
    axes[i].plot( np.average(U@Vh[:,Km.labels_ == i],axis=1)  ) 


# %% kernel retrieval
from numpy.linalg import pinv
ks = []
ws = []
for i in range(n_events):
    P = B_hat_lr[n_lags*i:n_lags*(i+1),:]
    k = np.average(P,axis=1)[:,np.newaxis]
    k = k / k.max()
    w = pinv(k)@P
    
    ks.append(k)
    ws.append(w)

W_hat = np.concatenate(ws)


# %%
K_hat = B_hat_lr @ pinv(W_hat)
fig, axes = plt.subplots()
axes.plot(K_hat)
# %%
Y_hatat = X @ K_hat @ W_hat 


yscl = 1.0
fig, axes = plt.subplots()

for i in range(Y.shape[1]):
    axes.plot(Y[:,i] * yscl + i, color='k', lw=1)

for i in range(Y.shape[1]):
    axes.plot(Y_hat[:,i] * yscl + i, color='r', lw=1)

for i in range(Y.shape[1]):
    axes.plot(Y_hatat[:,i] * yscl + i, color='c', lw=1)
# %%
