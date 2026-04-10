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
different events influence the same latent
one event might influence multiple latents
event2latent_map
"""

# %% make L
n_samples = 1000
dt = 1 
tvec = np.arange(0, n_samples*dt, dt)
n_events = 6
n_latents = 5 # = true rank

# now called W_k
w_events_2_latents = np.random.rand(n_events, n_latents)
W_K = w_events_2_latents

# timestamps are the event occurences
rates = np.random.rand(n_events) / 5
true_times = []
for rate in rates:
    tstamps = poisson_process(1/rate, t_min=0, t_max=n_samples*dt)
    true_times.append(tstamps)

# kernels
kvec = np.linspace(-49,50,100)
kernels = []
mus = np.random.randn(n_events) * 5
sigs = (np.random.rand(n_events)+0.1)*10

for i in range(n_events):
    kernels.append(kernel_gen(kvec, mus[i], sigs[i]))

# forming the latents
L = np.zeros((n_samples, n_latents))
for i in range(n_events):
    for j in range(n_latents):
        w = w_events_2_latents[i,j]

        L[:,j] += w * np.convolve(binarize(tvec, true_times[i]), kernels[i], mode='same')

# %% plot kernels
fig, axes = plt.subplots(ncols=n_events,sharey=True)
for i in range(n_events):
    axes[i].plot(kernels[i])

# %% make W (now called W_L)
true_rank = n_latents
n_features = 100

loc = 1
scale = 1
density = 0.3
dist = stats.distributions.norm(loc=loc,scale=scale)
W_L = sparse.random(true_rank, n_features,  density=density, data_rvs=dist.rvs).toarray()
W_L = np.random.randn(size=(true_rank, n_features))
# %% THE DATA GENERATION
Y = L @ W_L
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

n_lags = 100
X = lag_combine(tvec, true_times[:n_events], n_lags)

# make K
K = np.zeros((n_lags * n_events, n_events))
for i in range(n_events):
    K[n_lags*i:n_lags*(i+1),i] = kernels[i]

plt.plot(K)

# %% for now, just set rank
r = true_rank

# %% for now just ridge, much faster
lam = ridge_lambda_est_CV(LM, Y, X, args=(), n_samples=500)
B_hat = LM(Y, X, lam)

# lower rank
U, S, Vh = reduce_rank(*SVD(B_hat),r, return_matrix=True)
B_hat_lr = U @ S @ Vh

Y_hat = X @ B_hat
Y_hat_lr = X @ B_hat_lr

# %% kernel retrieval - no sub kernel variant
# splitting SVD from B_hat
ks = []
ws = []
# stol = 0.5 # 
r = true_rank

K_hat = np.zeros((n_lags*n_events, n_events))

for i in range(n_events):
    # splitting B_hat into each set of lagged regressors
    P = B_hat_lr[n_lags*i:n_lags*(i+1),:]

    # 
    U, s, Vh = SVD(P, return_matrix=False)
    k = U[:, 0][:,np.newaxis]

    # norm: peak = pos
    if k[np.argmax(np.absolute(k))] < 0:
        k = k * -1

    # weights
    K_hat[n_lags*i:n_lags*(i+1),i] = k[:,0]

plt.plot(K_hat)
W_KL_hat = linalg.pinv(K_hat) @ B_hat_lr
# <- this looks generally better than the "measured version"
# could be a normalization issue


"""
things that I know about 
W = W_K @ W_L
W can be measured: matrix "events x cells" - average activity

W_K is (n_events x n_latents)
W_L is (n_latents x n_features) (=cells) n_latents is known

n_latents are known
no assumptions about orthogonality, sign ... sparsity ...


"""


# %% digging into W_KL can be measured
# W_KL_hat
W_KL_hat = np.zeros((n_events, n_features))
for i in range(n_events):
    for j in tqdm(range(n_features)):
        # for t_ix, t in enumerate(true_times):
        for t_ix in times2inds(tvec, true_times[i]):
            if t_ix - int(n_lags/2) > 0 and t_ix + int(n_lags/2) < n_samples-1:
                W_KL_hat[i,j] += np.sum(Y[t_ix - int(n_lags/2) : t_ix + int(n_lags/2),j])
            

# %%
matshow(W_KL_hat)
matshow(W_K@W_L)

# %%
K_hatn = B_hat_lr @ linalg.pinv(W_KL_hat)
plt.figure()
plt.plot(K_hatn)

# %%
K_hat_n = np.zeros(K_hatn.shape)
for i in range(n_events):
    K_hat_n[n_lags*i:n_lags*(i+1),i] = K_hatn[n_lags*i:n_lags*(i+1),i]

plt.figure()
plt.plot(K_hat_n)




# %% FROM HERE ON - injecting the code to get W_K_hat by fitting

# %% monte carlo esque
from scipy.optimize import minimize

# def LM(Y, X, alpha=0):
#     """ (multiple) linear regression with regularization """
#     # ridge regression
#     I = np.diag(np.ones(X.shape[1]))
#     B_hat = linalg.pinv(X.T @ X + alpha *I) @ X.T @ Y # ridge regression
#     Y_hat = X @ B_hat
#     return B_hat

# def Rss(Y, Y_hat):
#     return np.sum((Y-Y_hat)**2)

def l2(Y,Y_hat):
    return np.sum((Y-Y_hat)**2)

# def obj_fun_Lasso(w, shape, W_KL, alpha):
#     W_K_hat = np.reshape(w, shape)
#     lasso = linear_model.Ridge(alpha=alpha)
#     lasso.fit(W_K_hat, W_KL)
#     W_L_hat = lasso.coef_.T
#     W_KL_hat = W_K_hat @ W_L_hat
#     return Rss(W_KL_hat, W_KL)**2

def obj_fun_LM(w, shape, W_KL, alpha):
    W_K_hat = np.reshape(w, shape)
    W_L_hat = LM(W_KL, W_K_hat, alpha)
    W_KL_hat = W_K_hat @ W_L_hat
    return l2(W_KL_hat, W_KL)

def single_run(W_K_hat0, W_KL, alpha):
    # alpha = ridge_lambda_est_CV(LM, W_KL, W_K_hat0)
    res = minimize(obj_fun_LM, W_K_hat0.flatten(), args=(W_K_hat0.shape, W_KL, alpha))
    W_K_hat = np.reshape(res.x, W_K_hat0.shape)
    return W_K_hat

def multi_run(N, sigma, W_K_hat0, W_KL, alpha):
    W_K_hats = []
    for i in tqdm(range(N)):
        W_K_hat0_ = W_K_hat0 + np.random.randn(*W_K_hat0.shape) * sigma 
        W_K_hat = single_run(W_K_hat0_, W_KL, alpha)
        W_K_hats.append(W_K_hat)
    return W_K_hats

def eval_K_hat(W_K_hat, W_KL):
    W_L_hat = linalg.pinv(W_K_hat) @ W_KL
    W_KL_hat = W_K_hat @ W_L_hat
    return Rss(W_KL, W_KL_hat)

# %% ini
from scipy.spatial.distance import euclidean

W_KL = W_K @ W_L # "true" To be replaced with W_KL_hat!
W_K_hat0 = np.random.randn(*W_K.shape)
min_err = eval_K_hat(W_K_hat0, W_KL)

# W_K_hat0 = W_K + np.random.randn(*W_K.shape) * 0.5

min_errs = []
errs = []

print(euclidean(W_K.flatten(), W_K_hat0.flatten()))

# %% run
N = 20
M = 60
W_K_hat_best = W_K_hat0
errs_all = []
sigs = np.linspace(0.1,0.001,M)
for j in range(M):
    W_K_hats = multi_run(N, sigs[j], W_K_hat_best, W_KL, 0.0)
    errs = []
    for i, W_K_hat in enumerate(W_K_hats):
        errs.append(eval_K_hat(W_K_hat, W_KL))
    errs_all.append(errs)
    
    # only update if improvement - dangerous?
    if np.min(errs) < min_err:
        min_err = np.min(errs)
        W_K_hat_best = W_K_hats[np.argmin(errs)]

    # min_errs.append(min_err)

# %% plot the errors over its
Errs_all = np.array(errs_all)
fig, axes = plt.subplots()
plt.plot(Errs_all)
plt.plot(np.min(Errs_all,axis=1),lw=2,color='r')

# %% visual mat compare 
print(euclidean(W_K.flatten(), W_K_hat_best.flatten()))

fig, axes = plt.subplots(ncols=2)
matshow(W_K, axes=axes[0])
matshow(W_K_hat_best, axes=axes[1])





# %% exploring different optimization algorithms
# basinhopping
from scipy import optimize
alpha = 0
kwargs = dict(args=(W_K_hat0.shape, W_KL, alpha))
res = optimize.basinhopping(obj_fun_LM, W_K_hat0.flatten(), minimizer_kwargs=kwargs, stepsize=0.2, niter=200)
W_K_hat = np.reshape(res.x, W_K_hat0.shape)
print(euclidean(W_K.flatten(), W_K_hat.flatten()))

fig, axes = plt.subplots(ncols=2)
matshow(W_K, axes=axes[0])
matshow(W_K_hat, axes=axes[1])

# %% diff evo
args = (W_K_hat0.shape, W_KL, alpha)
bounds = ((-5,5),) * W_K_hat0.flatten().shape[0]
res = optimize.differential_evolution(obj_fun_LM, args=args, bounds=bounds)
W_K_hat = np.reshape(res.x, W_K_hat0.shape)
print(euclidean(W_K.flatten(), W_K_hat.flatten()))

fig, axes = plt.subplots(ncols=2)
matshow(W_K, axes=axes[0])
matshow(W_K_hat, axes=axes[1])

# %% direct
args = (W_K_hat0.shape, W_KL, alpha)
bounds = ((-5,5),) * W_K_hat0.flatten().shape[0]
res = optimize.direct(obj_fun_LM, args=args, bounds=bounds, locally_biased=False)
W_K_hat = np.reshape(res.x, W_K_hat0.shape)

fig, axes = plt.subplots(ncols=2)
matshow(W_K, axes=axes[0])
matshow(W_K_hat, axes=axes[1])

# %% dual annealing
args = (W_K_hat0.shape, W_KL, alpha)
bounds = ((-5,5),) * W_K_hat0.flatten().shape[0]
res = optimize.dual_annealing(obj_fun_LM, args=args, bounds=bounds)
W_K_hat = np.reshape(res.x, W_K_hat0.shape)

fig, axes = plt.subplots(ncols=2)
matshow(W_K, axes=axes[0])
matshow(W_K_hat, axes=axes[1])









# %%
# W_K_hat = W_K_hat_best
# W_K_hat = W_K_hat0

# %%
W_L_hat = linalg.pinv(W_K_hat) @ W_KL
fig, axes = plt.subplots(ncols=2)
matshow(W_L, axes=axes[0])
matshow(W_L_hat, axes=axes[1])

W_KL_hat = W_K_hat @ W_L_hat
fig, axes = plt.subplots(ncols=2)
matshow(W_KL, axes=axes[0])
matshow(W_KL_hat, axes=axes[1])


# %% L has orthogonal columns
matshow(linalg.pinv(L) @ L)

# %% does L_hat?
L_hat = X @ K_hat @ W_K_hat
matshow(linalg.pinv(L_hat) @ L_hat)






























# %% can I formulate this as a LM

# U, s, Vh = reduce_rank(*SVD(W_KL_hat), n_latents, return_matrix=True)
# matshow(Vh)
# matshow(L.T)

def obj_func(U, shape, Y, X, K_hat):
    W_r = np.reshape(U,shape)
    # W_r = np.random.rand(n_events, n_latents)
    B_hat_r = LM(Y, X@K_hat@W_r, lam=0)
    return Rss(Y, X@K_hat@W_r@B_hat_r) + np.sum(B_hat_r**2)

w0 = np.random.rand(n_events, n_latents)

res = minimize(obj_func,w0.flatten(), args=(w0.shape, Y, X, K_hat))

# %%
W_L_hato = np.reshape(res.x,w0.shape)
matshow(W_L_hato)
matshow(W_K)


# %%
WkWl = linalg.pinv(K_hat)@B_hat_lr
matshow(WkWl)

# %% 
U, s, Vh = reduce_rank(*SVD(WkWl),5, return_matrix=False)
matshow(U)

# %%
# W_hat = np.concatenate(ws)
import sys
sys.path.append("/home/georg/code/MEP-Orthogonal-NMF/")
from MEP_ONMF import ONMF_DA
W_k_hat, W_l_hat, model = ONMF_DA.func(WkWl.T, r)
matshow(W_k_hat)
matshow(W_l_hat)

# %%
matcompare(W, W_k_hat.T)

# %%
