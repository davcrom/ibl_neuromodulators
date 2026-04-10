# %% imports
import numpy as np
from numpy import linalg

from scipy.linalg import diagsvd
from scipy.optimize import minimize

from tqdm import tqdm

# https://stackoverflow.com/questions/32612180/eliminating-warnings-from-scikit-learn
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn import linear_model
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

# general helpers

def flatten(As):
    """ parameter packing helper """
    return np.concatenate([A.flatten() for A in As]), [A.shape for A in As]

def unflatten(A, shapes):
    """ unpacking helper
    the inverser of flatten() """

    ix = np.cumsum([np.prod(shape) for shape in shapes])
    U = np.array_split(A, ix)[:-1]
    U = [U[i].reshape(shapes[i]) for i in range(len(U))]
    return U


# SVD related

def SVD(A, return_matrix=True):
    # helper
    U, s, Vh = linalg.svd(A)
    if return_matrix:
        S = diagsvd(s, U.shape[0], s.shape[0])
        return U, S, Vh
    else:
        return U, s, Vh

def reduce_rank(U, s, Vh, r, return_matrix=True):
    # helper, tolerant to s type being direct output of SVD or
    # S
    if len(s.shape) > 1:
        S = s
    else:
        S = diagsvd(s, U.shape[0], s.shape[0])

    U_lr = U[:, :r]
    Vh_lr = Vh[:r, :]

    if return_matrix:
        S_lr = S[:r, :r]
        return U_lr, S_lr, Vh_lr
    else:
        s_lr = s[:r]
        return U_lr, s_lr, Vh_lr


def Rss(Y, Y_hat, normed=True):
    # == Frobenius norm
    """ evaluate (normalized) model error """
    e = Y_hat - Y

    if len(e.shape) == 1:
        e = e[:, np.newaxis]

    Rss = np.trace(e.T @ e)
    if normed:
        Rss /= Y.shape[0]
    return Rss

# linear models
def LM(Y, X, lam):
    """ closed form solution ridge regression """

    # ridge regression
    I = np.diag(np.ones(X.shape[1]))
    B_hat = linalg.pinv(X.T @ X + lam*I) @ X.T @ Y

    # Y_hat = X @ B_hat
    return B_hat

def LM_enet(Y, X, alpha=1, l1_ratio=0.5):
    """ elastic net regression - sklearn implementation """
    model = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X, Y)
    B_hat = model.coef_.T
    return B_hat

def LM_rr_direct(Y, X, lam, r):
    # https://dept.stat.lsa.umich.edu/~jizhu/pubs/Mukherjee-SADM11.pdf
    # closed form solution!

    """
    Vh = Unitary matrix having right singular vectors as rows.
    Of shape (N, N) or (K, N) depending on full_matrices.

    """

    # Vh_lr = reduce_rank(*SVD(Y),5)[2]

    # Pr
    Vh = linalg.svd(Y)[2]
    Pr = np.zeros((Y.shape[1], Y.shape[1]))
    for i in range(r):
        Pr  += Vh[i,:][:,np.newaxis] * Vh[i,:][:,np.newaxis].T

    # ridge regression
    # I = np.diag(np.ones(X.shape[1]))
    # B_hat = linalg.pinv(X.T @ X + lam*I) @ X.T @ Y @ Pr

    B_hat_lr = LM(Y, X, lam) @ Pr
    return B_hat_lr

def LM_rr_post(Y, X, lam, r):
    B_hat = LM(Y, X, lam)
    U,S,Vh = reduce_rank(*SVD(B_hat),r, return_matrix=True)
    B_hat_lr = U @ S @ Vh
    return B_hat_lr

def LM_rr_post_enet(Y, X, lam, r, rho):
    B_hat = LM_enet(Y, X, lam, rho)
    U,S,Vh = reduce_rank(*SVD(B_hat),r, return_matrix=True)
    B_hat_lr = U @ S @ Vh
    return B_hat_lr

# XVAL
def LM_CV(LM_fun, Y, X, args, cv=5):
    kf = KFold(n_splits=cv)
    splits = list(kf.split(X, Y))
    
    res = []
    B_hats = []
    for k in range(cv):
        ix_train, ix_test = splits[k]
        B_hat = LM_fun(Y[ix_train], X[ix_train], *args)
        Y_hat_test = X[ix_test] @ B_hat

        B_hats.append(B_hat)
        res.append(Rss(Y[ix_test], Y_hat_test))
    return B_hats, res

# plotter
def plot_compare_weights(B, B_hat, pc=95, axes=None):
    if axes is None:
        fig, axes = plt.subplots(ncols=3)

    s = np.percentile(B, pc)
    kwargs = dict(vmin=-s, vmax=s, cmap='PiYG')
    axes[0].matshow(B, **kwargs)
    axes[1].matshow(B_hat, **kwargs)
    axes[2].matshow(B - B_hat, **kwargs)
    return axes

# %% 
"""
 
 ########     ###    ########    ###        ######   ######## ##    ## 
 ##     ##   ## ##      ##      ## ##      ##    ##  ##       ###   ## 
 ##     ##  ##   ##     ##     ##   ##     ##        ##       ####  ## 
 ##     ## ##     ##    ##    ##     ##    ##   #### ######   ## ## ## 
 ##     ## #########    ##    #########    ##    ##  ##       ##  #### 
 ##     ## ##     ##    ##    ##     ##    ##    ##  ##       ##   ### 
 ########  ##     ##    ##    ##     ##     ######   ######## ##    ## 
 
"""
from scipy import sparse
import scipy.stats as stats

def data_gen(n_features, n_samples, n_regressors, true_rank, noise_sig=0, is_sparse=False, cov=False, sparse_params=None, cov_params=None):
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

# %%
def ridge_lambda_est_CV(model, Y, X, args=None):
    """ model agnostic lambda cv estimator """
    def obj_func(lam, Y, X, *args):
        B_hats, res = LM_CV(model, Y, X, (lam, *args), cv=5)
        return np.average(res)

    p0 = np.array([1])
    res = minimize(obj_func, p0, args=(Y, X, *args), bounds=((1e-24, None),), options=dict(disp=False))
    lam = res.x[0]
    return lam

def model_eval(model, Y, X, args=None):

    lam = ridge_lambda_est_CV(model, Y, X, args=args)
    out = {}
    
    # print("xval lambda for ridge: %.4f" % lam)

    B_hat = model(Y, X, lam, *args)
    out['full'] = Rss(Y, X@B_hat)
    # print("model without CV: %.4f" % Rss(Y, X@B_hat))

    # cv model performace
    B_hats, res = LM_CV(model, Y, X, (lam, *args), cv=5)
    out['cv_avg'] = np.average(res)
    # print("cv average model Rss: %.4f" %np.average(res))

    # averaged cv model
    B_hat = np.average(np.stack(B_hats, axis=2),axis=2)
    out['cv_pred'] = Rss(Y, X@B_hat)

    # print("xval lambda for ridge: %.4f" % lam)
    return out

# %%
data_dict = dict(
    n_features = 100, # 'm' = number of neurons
    n_samples = 200, # 'n' = number of time samples
    n_regressors = 50, # 'l' = number of all lags x events
    true_rank = 20, # 'r'
    noise_sig = 5.0
    )

sparse_dict = dict(loc=1,
                   scale=2,
                   density=0.3)

cov_dict = dict(density=0.2)

Y, X, B = data_gen(**data_dict, is_sparse=True, sparse_params=sparse_dict, cov=True, cov_params=cov_dict)

# %%
import seaborn as sns
fig, axes = plt.subplots()

n_cv = 5
rr = np.arange(3,15)
models = [LM_rr_direct, LM_rr_post]
colors = sns.color_palette('tab10', n_colors=len(models))
colors = dict(zip(models,colors))


for model in models:
    ress = np.zeros((rr.shape[0], n_cv))
    for i, r in enumerate(tqdm(rr)):
        lam = ridge_lambda_est_CV(model, X, X, args=(r,))
        B_hats, res = LM_CV(model, Y, X, (lam, r), cv=n_cv)
        ress[i,:] = res


    mid = np.average(ress, axis=1)
    above = mid + np.std(ress, axis=1)
    below = mid - np.std(ress, axis=1)

    axes.fill_between(rr, below, above, color=colors[model], alpha=0.2)
    axes.plot(rr, mid, color=colors[model], label=model.__name__)
    
axes.legend()

lam = ridge_lambda_est_CV(LM, Y, X, args=())
B_hats, res = LM_CV(LM, Y, X, (lam, ), cv=5)
axes.axhline(np.average(res), linestyle=':', color='k')

# %%
# def obj_func(p, Y, X):
#     lam, rho = p
#     B_hats, res = LM_CV(LM_enet, Y, X, (lam, rho), cv=5)
#     return np.average(res)

# p0 = np.array([0.5, 0.5])
# res = minimize(obj_func, p0, args=(Y, X),bounds=((1e-10, 1e-10), (None, 1)), options=dict(disp=False))
# lam_e, rho_e = res.x

# %%
B_hats, res = LM_CV(LM_enet, Y, X, (lam_e, rho_e), cv=5)
axes.axhline(np.average(res), linestyle=':', color='r')

# %% SVD from the B_hats is not the same!
B_hat_post = LM_rr_post(Y, X, lam, r)
B_hat_direct = LM_rr_direct(Y, X, lam, r)

plot_compare_weights(B_hat_post, B_hat_direct)

fig, axes = plt.subplots(ncols=3, nrows=3)
svd_direct = SVD(B_hat_direct)
svd_post = SVD(B_hat_post)
for i in range(3):
    plot_compare_weights(svd_direct[i], svd_post[i], axes=axes[i,:])








# %%
