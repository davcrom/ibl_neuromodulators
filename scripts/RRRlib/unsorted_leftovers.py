# %% imports
import numpy as np
from numpy import linalg

from scipy import sparse
import scipy.stats as stats
from tqdm import tqdm

from RRRlib import *

import matplotlib.pyplot as plt

def data_gen(n_features, n_samples, n_regressors, true_rank, noise_sig=0, is_sparse=False, cov=False, sparse_params=None, cov_params=None):
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
mu = 0
sig = 0.7
kvec = np.linspace(-5,5,100)
K = stats.distributions.norm(mu, sig).pdf(kvec)
K = K / np.sum(K)

n_samples = 5000 # 'n' = number of time samples
n_events = 5

y = np.zeros((n_samples, n_events))
tt = []
for i in range(n_events):
    bvec = np.zeros(n_samples)
    true_times = np.random.randint(0, n_samples, 50)
    tt.append(true_times)
    bvec[true_times] = 1
    y[:,i] = np.convolve(bvec ,K, mode='same')


# y[:,0] += y[:,1]

# %%
n_features = 100 # 'm' = number of neurons
w_units = np.random.randn(n_events, n_features)+1

data_rvs = stats.distributions.norm(loc=1,scale=1).rvs
w_units = sparse.random(n_events, n_features, density=0.1, data_rvs=data_rvs).toarray()

fig, axes = plt.subplots()
axes.matshow(w_units)
axes.set_aspect('auto')

Y = y @ w_units
noise_sig = 0.10
Y = Y + np.random.randn(*Y.shape) * noise_sig

X = lag_combine(tvec, tt, n_lags)

yscl = 1.5
fig, axes = plt.subplots()
for i in range(Y.shape[1]):
    axes.plot(Y[:,i] * yscl + i, color='k', lw=1)

# %% -> LRlib.py
"""
timestamps of Y samples
timestamps of event
-> binarized event vector

time2ind
times2inds

timestamps -> lagged reg
make_model_matrix(regs)
"""

def times2inds(tvec, times):
    return [np.argmin(np.absolute(tvec - t)) for t in times]

def binarize(tvec, times):
    bvec = np.zeros(tvec.shape[0])
    bvec[times2inds(tvec, times)] = 1
    return bvec

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


# %% make a model matrix
dt = 1
tvec = np.arange(0, Y.shape[0], dt)
l = binarize(tvec, tt[0])

n_lags = 50
X = lag_regressor(l, n_lags)

# %% make a kernel
mu = 0
sig = 0.7
kvec = np.linspace(-5,5,50)
K = stats.distributions.norm(mu, sig).pdf(kvec)
K = K / np.sum(K)

L = np.convolve(l, K, mode='same')
plt.plot(L)

# %%
plt.plot(LM(L,X,lam=0))
plt.plot(K)


# %%
X = lag_combine(tvec, [tt[0],tt[1]], 50)
L0 = np.convolve(binarize(tvec, tt[0]), K, mode='same')
L1 = np.convolve(binarize(tvec, tt[1]), K, mode='same')

L = np.stack([L0,L1]).T

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
