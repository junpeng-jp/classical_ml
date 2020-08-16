import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.base import clone

def clusterplot_2D(X1, X2, title="Scatterplot", xlabel="X1", ylabel="X2", ax=None, **plot_kwargs):
    if not ax:
        ax = plt.gca()

    ax.scatter(X1, X2, **plot_kwargs)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def decisionplot_2D(X, classifier, ax=None, c=None, cmap=None):
    X = X.copy()
    if not ax:
        ax = plt.gca()

    if X.shape[1] != 2:
        raise AttributeError('X should only have 2 dimensions for a 2-D plot.')
        
    xx, yy = make_meshgrid(X[:, 0], X[:, 1])
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, cmap = cmap, alpha = 0.2, linestyles='dashed')

    clusterplot_2D(
        X[:, 0], X[:, 1],
        title = "Decision Boundary Plot",
        ax = ax, c = c, cmap = cmap)


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy