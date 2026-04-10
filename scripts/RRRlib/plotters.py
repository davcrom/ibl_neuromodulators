# %%
import matplotlib.pyplot as plt
import numpy as np

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

# plotter
def matcompare(A, B, pc=95, axes=None):
    if axes is None:
        fig, axes = plt.subplots(ncols=3)

    s = np.percentile(B, pc)
    kwargs = dict(vmin=-s, vmax=s, cmap='PiYG')
    axes[0].matshow(A, **kwargs)
    axes[1].matshow(B, **kwargs)
    axes[2].matshow(A - B, **kwargs)
    [ax.set_aspect('auto') for ax in axes]
    return axes

def matshow(A, axes=None, **kwargs):
    """ extent to percentile vmin, vmax, default to PiYG when neg and pos,
     zero centered """
    if axes is None:

        fig, axes = plt.subplots()
    axes.matshow(A, **kwargs)
    axes.set_aspect('auto')
    # name = f'{A=}'.split('=')[0]
    # axes.set_title(name)

    return axes

def plot_stacked_lines(Y, yscl=1, axes=None, **kwargs):
    if axes is None:
        fig, axes = plt.subplots()

    line_kwargs = dict(color='k', lw=1)
    line_kwargs.update(kwargs)
    for i in range(Y.shape[1]):
        axes.plot(Y[:,i] * yscl + i, **line_kwargs)

    return axes

def plot_kernels(kernels):
    fig, axes = plt.subplots(ncols=len(kernels))
    for i in range(len(kernels)):
        axes[i].plot(kernels[i])
