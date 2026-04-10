"""
if W_K is 
how kernel maps onto latents
shape (n_kernels = n_events, n_latents) and n_events > n_events
and 
W_K is upper triangular

one event can influence multiple latents
-> multiple non-zero per row

multiple events influence one latent
-> multiple non-zero per column

as soon as one out of diagonal entry is non-zero both conditions are met
not if shape is not square (first dim > second dim)

not clear currently if those are good assumptions
water droplet consumption (an event)
leads to change in reward expectation and thirst (both latent)

large water droplet vs small water droplet
(both events) influence both thirst and rew exp latent

idea:
numerically find W_K under the assumption that W_L is sparse
min: wrt W_K
Rss(W, W_Hat)
W_hat = W_K @ W_L_hat
W_L_hat : lasso on (Y=W_hat, X=W_K)
"""

# %%
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import scipy as sp
from tqdm import tqdm

# %%
n_events = 10
n_kernels = 5
n_features = 20
r = 5

# %% row only
# W_K = np.zeros((n_events, n_kernels))

# for i in range(n_kernels):
#     W_K[i,i] = 1

# for i in range(n_kernels, n_events):
#     j = np.random.randint(n_kernels)
#     W_K[i,j] = 1

# fig, axes = plt.subplots()
# axes.matshow(W_K)

# %% random W_K, sparse W_L
W_K = np.random.randn(n_events, n_kernels)
dist = sp.stats.distributions.norm(loc=0,scale=1)
W_L = sp.sparse.random(n_kernels, n_features, density=0.3, data_rvs=dist.rvs).toarray()

W_KL = W_K@W_L
# plt.matshow(W)
plt.matshow(W_K)
plt.matshow(W_L)
plt.matshow(W_KL)

# %% random W_K, random W_L
W_K = np.random.randn(n_events, n_kernels)
W_L = np.random.randn(n_kernels, n_features)

W_KL = W_K@W_L

plt.matshow(W_K)
plt.matshow(W_L)
plt.matshow(W_KL)


# %% monte carlo esque
from scipy.optimize import minimize

def LM(Y, X, alpha=0):
    """ (multiple) linear regression with regularization """
    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + alpha *I) @ X.T @ Y # ridge regression
    Y_hat = X @ B_hat
    return B_hat

def Rss(Y, Y_hat):
    return np.sum((Y-Y_hat)**2)

def obj_fun_Lasso(w, shape, W_KL, alpha):
    W_K_hat = np.reshape(w, shape)
    lasso = linear_model.Ridge(alpha=alpha)
    lasso.fit(W_K_hat, W_KL)
    W_L_hat = lasso.coef_.T
    W_KL_hat = W_K_hat @ W_L_hat
    return Rss(W_KL_hat, W_KL)**2

def obj_fun_LM(w, shape, W_KL, alpha):
    W_K_hat = np.reshape(w, shape)
    # TODO here - cv alpha
    W_L_hat = LM(W_KL, W_K_hat, alpha)
    W_KL_hat = W_K_hat @ W_L_hat
    return Rss(W_KL_hat, W_KL)**2

def single_run(W_K_hat0, W_KL, alpha):
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
W_K_hat0 = np.random.randn(*W_K.shape)
min_err = eval_K_hat(W_K_hat0, W_KL)
# W_K_hat0 = W_K #

min_errs = []
errs = []

# %% run
N = 10
M = 20
W_K_hat_best = W_K_hat0
errs_all = []
for j in range(M):
    W_K_hats = multi_run(N, 0.2, W_K_hat_best, W_KL, 0.01)
    errs = []
    for i, W_K_hat in enumerate(W_K_hats):
        errs.append(eval_K_hat(W_K_hat, W_KL))
    errs_all.append(errs)
    
    # only update if improvement - dangerous?
    if np.min(errs) < min_err:
        min_err = np.min(errs)
        W_K_hat_best = W_K_hats[np.argmin(errs)]

    # min_errs.append(min_err)

# %% 
Errs_all = np.array(errs_all)
fig, axes = plt.subplots()
kwargs = dict( zip(('vmin','vmax'), np.percentile(Errs_all, (5,55))))
axes.matshow(Errs_all,**kwargs)

# %%
fig, axes = plt.subplots()
plt.plot(np.min(Errs_all,axis=1))



# %% eval
for i, W_K_hat in enumerate(W_K_hats):
    W_L_hat = linalg.pinv(W_K_hat) @ W_KL
    W_KL_hat = W_K_hat @ W_L_hat
    errs.append(Rss(W_KL, W_KL_hat))

plt.plot(errs,'.')
# %%
fig, axes = plt.subplots()
axes.plot(min_errs,'.')
# axes.set_ylim(1e-6,1e-7)
# %%
W_K_hat = W_K_hat_best
fig, axes = plt.subplots(ncols=2)
axes[0].matshow(W_K)
axes[1].matshow(W_K_hat)

W_L_hat = linalg.pinv(W_K_hat) @ W_KL
fig, axes = plt.subplots(ncols=2)
axes[0].matshow(W_L)
axes[1].matshow(W_L_hat)

W_KL_hat = W_K_hat @ W_L_hat
fig, axes = plt.subplots(ncols=2)
axes[0].matshow(W_KL)
axes[1].matshow(W_KL_hat)


# %%
