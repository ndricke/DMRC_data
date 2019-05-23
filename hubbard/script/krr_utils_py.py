"""
Helper functions for kernel ridge regression (KRR)
"""


import numpy as np


def rbf_kernel(x1, x2, sigma):
    """Radial basis function, K(x1,x2,sigma) = exp[-|x1-x2|^2/(2*sigma)]
    """
    assert(x1.shape == x2.shape)
    return np.exp(-np.linalg.norm(x1-x2)**2. / 2. / sigma**2.)


def rbf_kernel_batch(X1, X2, sigma):
    """Same as above, but X1 and X2 are both data matrix
    """
    assert(X1.shape[0] == X2.shape[0])
    d, N1 = X1.shape
    N2 = X2.shape[1]
    dist2 = np.zeros([N1,N2])
    for i in range(N1):
        dist2[i,:] = np.linalg.norm(X1[:,i] - X2.T, axis=1)**2.

    return np.exp(-dist2/2./sigma**2.)


def get_rbf_J(alp, K, Xtrain):
    """J_{i,mu} = sum_n alp_n * K(X[:,i], Xtrain[:,n]) * Xtrain[mu,n]

    Notes:
        This quantity is needed for computing rbf derivative.
    """
    return np.einsum("n,in,mn->im", alp, K, Xtrain, optimize=True)


def krr(X, t, lbd, kernel, params):
    """A funciton that performs KRR

    Inp:
        X (np.array, d*N):
            Data matrix; each column represents a data point
        t (np.array, N):
            Target values
        lbd (float):
            Regularization constant
        kernel (function):
            K[i,j] = kernel(X[:,i], X[:,j], *params) is the (i,j)-element of
            the kernel matrix
        params (tuple):
            Extra parameters that are needed for calling the kernel function

    Out:
        alp (np.array, N):
            Dual-space coefficents such that the prediction is
                y(x) = sum_n^N kernel(x, X[:,n]) * alp[n]
    """
    d, N = X.shape
    assert(t.size == N)

    # construct the kernel matrix
    if kernel == rbf_kernel:
        K = np.zeros([N,N])
        for i in range(N):
            for j in range(i,N):
                K[i,j] = K[j,i] = kernel(X[:,i], X[:,j], *params)
    elif kernel == rbf_kernel_batch:
        K = rbf_kernel_batch(X, X, *params)
    else:
        raise RuntimeError("Unknown kernel!")

    # solving the kernel equation
    alp = np.linalg.solve(K + lbd*np.eye(N), t)

    return alp


def krr_prediction(x, alp, X, kernel, params):
    """KRR prediction on input data

    Inp:
        x (np.array, d):
            Input data vector
        alp (np.array, N):
            Dual-space coefficients from training
        X (np.array, d*N):
            Training data matrix
        kernel (func):
            Kernel function
        params (tuple):
            Extra parameters that the kernel function takes

    Out:
        y (float):
            A scalar output
    """
    d, N = X.shape
    y = np.array([kernel(x, X[:,i], *params) for i in range(N)]) @ alp

    return y


def krr_prediction_batch(Xt, alp, X, kernel, params):
    Nt = Xt.shape[1]
    if kernel == rbf_kernel:
        tt = np.array([krr_prediction(Xt[:,i], alp, X, kernel, params) \
            for i in range(Nt)])
    elif kernel == rbf_kernel_batch:
        tt = kernel(Xt, X, *params) @ alp

    return tt


def get_random_partition(Ntot, Ns):
    n = len(Ns)
    assert(sum(Ns) == Ntot)
    a = list(range(Ntot))
    np.random.shuffle(a)

    rets = []
    ind = 0
    for i in range(n):
        rets.append(list(a[ind:ind+Ns[i]]))
        ind += Ns[i]

    return rets


if __name__ == "__main__":
    # A sinusoidal curve with Gaussian noise
    N = 6
    lbd = 1.E-7
    sigma = 0.1
    sigma_rbf = 3
    np.random.seed(17)
    # X = np.sort(np.random.rand(1,N) * np.pi * 2.)
    X = np.linspace(0,1.,N).reshape([1,N]) * 6.
    def func(x):
        return np.exp(-0.2*x) * np.cos(2.*x)
    t = func(X).reshape(N) + np.random.normal(0., sigma, N)
    alp = krr(X, t, lbd, rbf_kernel, (sigma_rbf,))
    xs = np.arange(0, 2.*np.pi, 0.1)
    ys = np.array([krr_prediction(np.array([xx]), alp, X, rbf_kernel, \
        (sigma_rbf,)) for xx in xs])
    ys_true = func(xs)
    from matplotlib import pyplot as plt
    import seaborn as sns
    sns.set()
    fig = plt.figure()
    ax = fig.gca()

    l1, = ax.plot(X.reshape(N), t, "o", c="#d62728")
    l2, = ax.plot(xs, ys, "-", c="#1f77b4", lw=2)
    l3, = ax.plot(xs, ys_true, "--", c="#2ca02c")
    llgs = [l1, l2, l3]
    legs = ["data", "KRR", "\"Ground-truth\""]

    labelfont = 18
    tickfont = 16

    ax.tick_params(axis="both", labelsize=tickfont)

    plt.tight_layout()
    plt.legend(llgs, legs, fontsize=tickfont-1, frameon=False, ncol=1)

    plt.savefig("krr_demo_{:d}.pdf".format(N))
    plt.close(fig)
