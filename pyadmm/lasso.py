__author__ = 'jr'

import numpy as np
import numpy.linalg as npl
import numpy.random as npr

import utils

import time


def factor(A, rho):
    m, n = A.shape;
    At = np.transpose(A)

    if m >= n:    # if skinny
        L = npl.cholesky(np.dot(At, A) + rho * np.eye(n));
    else:         # if fat
        L = npl.cholesky(np.eye(m) + 1 / rho * np.dot(A, At));

    U = np.transpose(L)

    return (L, U)


def shrinkage(x, kappa):
    z = np.maximum(0, x - kappa ) - np.maximum(0, -x - kappa )
    return z


def objective(A, b, lmbd, x, z):
    p = 1/2 * np.sum((np.dot(A, x) - b) ** 2) \
        + lmbd * npl.norm(z, 1);
    return p


def lasso(A, b, lmbd, rho, alpha):

    history = utils.init_history()
    t_start = time.clock()

    QUIET    = False
    MAX_ITER = 1000
    ABSTOL   = 1e-4
    RELTOL   = 1e-2

    m, n = A.shape;

    # save a matrix-vector multiply
    At = np.transpose(A)
    Atb = np.dot(At, b)

    x = np.zeros((n,1))
    z = np.zeros((n,1))
    u = np.zeros((n,1))

    # cache the factorization
    L, U = factor(A, rho)

    if not QUIET:
        print "%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n" % ("iter", "r norm", "eps pri", "s norm", "eps dual", "objective")

    for k in range(MAX_ITER):

        # x-update
        q = Atb + rho * (z - u) # temporary value
        if( m >= n ): # if skinny
            x = npl.solve(U, npl.solve(L, q))
        else: # if fat
            x = q/rho - np.dot(At, npl.solve(U, npl.solve(L, np.dot(A, q)))) / (rho ** 2)

        # z-update with relaxation
        zold = z;
        x_hat = alpha * x + (1 - alpha) * zold;
        z = shrinkage(x_hat + u, lmbd/rho);

        # u-update
        u = u + (x_hat - z);

        # diagnostics, reporting, termination checks
        objval = objective(A, b, lmbd, x, z);
        r_norm  = npl.norm(x - z);
        s_norm  = npl.norm(-rho*(z - zold));
        eps_pri = np.sqrt(n) * ABSTOL + RELTOL * np.maximum(npl.norm(x), npl.norm(-z));
        eps_dual= np.sqrt(n) * ABSTOL + RELTOL * npl.norm(rho*u);

        history["objval"].append(objval);
        history["r_norm"].append(r_norm)
        history["s_norm"].append(s_norm)
        history["eps_pri"].append(eps_pri)
        history["eps_dual"].append(eps_dual)

        if not QUIET:
            print "%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n" % (k, r_norm, eps_pri, s_norm, eps_dual, objval);

        if r_norm < eps_pri and s_norm < eps_dual:
            break;

    return (x, {})

