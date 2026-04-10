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
def LM_CV(model, Y, X, args, cv=5):
    kf = KFold(n_splits=cv)
    splits = list(kf.split(X, Y))
    
    res = []
    B_hats = []
    for k in range(cv):
        ix_train, ix_test = splits[k]
        B_hat = model(Y[ix_train], X[ix_train], *args)
        Y_hat_test = X[ix_test] @ B_hat

        B_hats.append(B_hat)
        res.append(Rss(Y[ix_test], Y_hat_test))
    return B_hats, res

def ridge_lambda_est_CV(model, Y, X, args=None, lam0=1, cv=5, n_samples=None):
    """ model agnostic lambda cv estimator """
    def obj_func(lam, model, Y, X, cv, *args):
        _, res = LM_CV(model, Y, X, (lam, *args), cv=cv)
        # return np.average(res)**2
        return np.sum(res)

    if args is None:
        args = ()

    if n_samples is not None:
        ix = np.random.randint(0,Y.shape[0],size=n_samples)
        Y = Y[ix]
        X = X[ix]

    lam0 = np.array([lam0])
    res = minimize(obj_func, lam0, args=(model, Y, X, cv, *args), bounds=((1e-10, None),), options=dict(disp=False))

    if not res.success:
        print("me")
        print(res.message)

    lam = res.x[0]
    return lam






# def model_eval(model, Y, X, args=None):

#     lam = ridge_lambda_est_CV(model, Y, X, args=args)
#     out = {}
    
#     # print("xval lambda for ridge: %.4f" % lam)

#     B_hat = model(Y, X, lam, *args)
#     out['full'] = Rss(Y, X@B_hat)
#     # print("model without CV: %.4f" % Rss(Y, X@B_hat))

#     # cv model performace
#     B_hats, res = LM_CV(model, Y, X, (lam, *args), cv=5)
#     out['cv_avg'] = np.average(res)
#     # print("cv average model Rss: %.4f" %np.average(res))

#     # averaged cv model
#     B_hat = np.average(np.stack(B_hats, axis=2),axis=2)
#     out['cv_pred'] = Rss(Y, X@B_hat)

#     # print("xval lambda for ridge: %.4f" % lam)
#     return out

# %%
