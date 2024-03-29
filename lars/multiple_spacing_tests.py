#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 16:35:42 2019
@author: Y. DE CASTRO, yohann.decastro "at" gmail
This py file contains functions for running multiple spacing testing on the LAR's knots
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy as sc
import random
from scipy.special import erfc
from scipy.stats import norm, binom

# %% Generate generate data

def generate_data(n, p, k, eta, A, X=None, theta0=None, S=None):
    '''
    We consider the linear model: y = X @ theta0 + W where
      - X is a (n x p) matrix drawn from N(0, Sigma) with Sigma_ij = eta^{|i-j|} with eta \in (0,1). Columns are normalized
      - theta0 has k non-zero coefficients. The non-zero coeffs are set uniformly at random to -A or +A
      - W is a standard gaussian vector
    '''
    if X is None:
        sig = np.ones(p)
        for i in range(1,p):
            sig[i] = eta * sig[i-1]
        Sigma = sc.linalg.circulant(sig)
        Sigma = np.tril(Sigma) + np.tril(Sigma, -1).T

        X = np.random.multivariate_normal(np.zeros(p), Sigma, size=n)
        for j in range(p):
            X[:,j] /= np.linalg.norm(X[:,j])
    if theta0 is None:
        S = random.sample(range(p), k)
        theta0 = np.zeros(p)
        theta0[S] = 2*(np.random.normal(0,1,k)>0)-1
        theta0 *= A
    y = X @ theta0 + np.random.normal(0,1,n)
    return theta0, X, S, y


# %% Generate variables
def generate_variables(predictors_number,
                       sample_size,
                       sparsity=0,
                       sigma=1,
                       noise_correlation=0,
                       normalization=False,
                       covariance_design=0):
    """Generate a linear model
    """

    p = predictors_number  # % vector size
    n = sample_size  # % sample size

    """ design
    """
    if np.all(covariance_design) == 0:
        X = norm.rvs(0, 1 / np.sqrt(n), [n, p])
    else:
        for k in range(0, n):
            X[k, ] = np.random.multivariate_normal(np.zeros(p), covariance_design)
    if normalization:
        X = np.dot(X, np.diag(np.dot(np.ones((1, np.shape(X)[0])), X * X)[0] ** (-0.5)))

    """ target vector
    """
    signs_beta_0 = 2 * binom.rvs(1, 0.5, size=sparsity) - 1  # % signs of sparse target vector
    beta_0 = np.concatenate([signs_beta_0, np.zeros(p - sparsity)])  # % sparse vector

    """ noise vector
    """
    if np.all(noise_correlation) == 0:
        noise_correlation = np.identity(n)
    epsilon = np.random.multivariate_normal(np.zeros(n),
                                            (sigma ** 2) * noise_correlation)  # % noise vector

    """ observation
    """
    Y = np.dot(X, beta_0) + epsilon

    """ whithening the observation
    """
    A = np.dot(np.matrix.transpose(X),
               np.linalg.inv(noise_correlation))
    R_bar = np.dot(A, X)
    Z = np.dot(A, Y)
    Z = np.concatenate([Z, -Z])
    R = np.block(
        [[R_bar, -R_bar],
         [-R_bar, R_bar]
         ])
    return [Z, R, sigma, X, noise_correlation, sparsity, beta_0, Y]


# %% Empirical Laws

def empirical_cdf(x):
    return (1 + np.argsort(np.argsort(x))) / len(x)


def empirical_law(predictors_number,
                  sample_size,
                  start,
                  end,
                  middle,
                  iterations=int(1e2),
                  display=0,
                  sparsity=0,
                  sigma=1,
                  noise_correlation=0,
                  normalization=False,
                  covariance_design=0,
                  eval_points=9973):  # 99991
    rep = 1
    result = []
    for i in range(0, iterations):
        Z, R, __, __, __, __, __, __ = generate_variables(predictors_number,
                                                          sample_size,
                                                          sparsity,
                                                          sigma,
                                                          noise_correlation,
                                                          normalization,
                                                          covariance_design)
        T = np.zeros(np.size(Z))
        lambdas = np.zeros(sample_size)
        indexes = np.zeros(sample_size)
        correls = np.zeros(sample_size)
        #        irrepresentable_check   = np.zeros(sample_size)
        var_R = R
        var_Z = Z

        for k in range(min(sample_size, predictors_number, np.linalg.matrix_rank(R, hermitian=True))):
            var_R, var_Z, T, lambdas[k], indexes[k], correls[k] = rec(var_R, var_Z, T)
        #            irrepresentable_check[k] = bool(np.amax(np.abs(T))<1)
        #        order_irrep = sum(np.cumprod(irrepresentable_check))
        lars = [lambdas, indexes, correls]
        alpha, __, __ = observed_significance_CBC(lars,
                                                  sigma,
                                                  start,
                                                  end,
                                                  middle,
                                                  eval_points,  # 99991,
                                                  rep)
        result += [alpha]

    if display:
        print("CBC: Significance of knot %s between knot %s and knot %s over %s iterations" % (
            middle, start, end, iterations))
        lispace_size = 2000
        x = np.linspace(0, 1, lispace_size)
        a = np.asarray(result)
        z = np.arange(0, 1, 1. / len(a))
        t = np.sort(a)
        # sns.set_style("white")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.step(t, z, color="lightcoral", lw=3)
        plt.plot(x, x, color="yellowgreen", linestyle="--")
        plt.legend([r'CDF of $\widehat\alpha_{%s,%s,%s}$' % (start, middle, end)], loc=2)
        plt.title("$p=$%s, $n=$%s over %s iterations" % (predictors_number, sample_size, iterations))
        plt.tight_layout()
        plt.savefig('law.png', format='png', dpi=600)
        plt.show()

    return result


def empirical_law_H0_H1(predictors_number,
                        sample_size,
                        start,
                        end,
                        middle,
                        iterations=int(1e2),
                        display=1,
                        sparsity=1,
                        sigma=1,
                        noise_correlation=0,
                        normalization=True,
                        covariance_design=0,
                        eval_points=9973):  # 99991,
    result0 = []
    result1 = []
    rep = 1
    for i in range(0, iterations):
        Z, R, __, __, __, __, __, __ = generate_variables(predictors_number,
                                                          sample_size,
                                                          sparsity=0,  # sparsity = 0
                                                          sigma=sigma,
                                                          noise_correlation=noise_correlation,
                                                          normalization=normalization,
                                                          covariance_design=covariance_design)
        T = np.zeros(np.size(Z))
        lambdas = np.zeros(sample_size)
        indexes = np.zeros(sample_size)
        correls = np.zeros(sample_size)
        #        irrepresentable_check   = np.zeros(sample_size)
        var_R = R
        var_Z = Z

        for k in range(min(sample_size, predictors_number, np.linalg.matrix_rank(R, hermitian=True))):
            var_R, var_Z, T, lambdas[k], indexes[k], correls[k] = rec(var_R, var_Z, T)
        #            irrepresentable_check[k] = bool(np.amax(np.abs(T))<1)
        #        order_irrep = sum(np.cumprod(irrepresentable_check))
        lars = [lambdas, indexes, correls]
        alpha0, __, __ = observed_significance_CBC(lars,
                                                   sigma,
                                                   start,
                                                   end,
                                                   middle,
                                                   eval_points,  # 99991,
                                                   rep)
        result0 += [alpha0]

    for i in range(0, iterations):
        Z, R, __, __, __, __, __, __ = generate_variables(predictors_number,
                                                          sample_size,
                                                          sparsity=sparsity,  # sparsity = 0
                                                          sigma=sigma,
                                                          noise_correlation=noise_correlation,
                                                          normalization=normalization,
                                                          covariance_design=covariance_design)
        T = np.zeros(np.size(Z))
        lambdas = np.zeros(sample_size)
        indexes = np.zeros(sample_size)
        correls = np.zeros(sample_size)
        #        irrepresentable_check   = np.zeros(sample_size)
        var_R = R
        var_Z = Z

        for k in range(min(sample_size, predictors_number, np.linalg.matrix_rank(R, hermitian=True))):
            var_R, var_Z, T, lambdas[k], indexes[k], correls[k] = rec(var_R, var_Z, T)
        #            irrepresentable_check[k] = bool(np.amax(np.abs(T))<1)
        #        order_irrep = sum(np.cumprod(irrepresentable_check))
        lars = [lambdas, indexes, correls]
        alpha1, __, __ = observed_significance_CBC(lars,
                                                   sigma,
                                                   start,
                                                   end,
                                                   middle,
                                                   eval_points,  # 99991,
                                                   rep)
        result1 += [alpha1]

    if display:
        print("CBC: Significance of knot %s between knot %s and knot %s over %s iterations" % (
            middle, start, end, iterations))
        lispace_size = 2000
        x = np.linspace(0, 1, lispace_size)
        a1 = np.asarray(result0)
        z1 = np.arange(0, 1, 1. / len(a1))
        t1 = np.sort(a1)
        a2 = np.asarray(result1)
        z2 = np.arange(0, 1, 1. / len(a2))
        t2 = np.sort(a2)
        #        sns.set_style("white")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.step(t1, z1, color="royalblue", lw=3)
        plt.step(t2, z2, color="lightcoral", lw=3)
        plt.plot(x, x, color="yellowgreen", linestyle="--")
        plt.legend([r'CDF of $\widehat\alpha_{%s,%s,%s}$ under $H_0$' % (start, middle, end),
                    r'CDF of $\widehat\alpha_{%s,%s,%s}$ under $H_1$' % (start, middle, end)],
                   loc=4)
        # plt.title ("$p=$%s, $n=$%s over %s iterations" %(predictors_number,sample_size, iterations))
        plt.tight_layout()
        plt.savefig('law_a-%s_b-%s_c-%s_sparisty-%s_CBC-%s_on_%s_at_%s.png' % (
            start, middle, end, sparsity, eval_points, time.strftime("%d_%m_%Y"), time.strftime("%H_%M_%S")),
                    format='png',
                    dpi=600)
        plt.show()

    return [result0, result1]


# %% Figure 1 and 2

def empirical_law_first_knots(predictors_number: int,
                              sample_size: int,
                              end: int,
                              start=0,
                              iterations=int(5e2),
                              display=0,
                              sparsity=0,
                              sigma=1,
                              noise_correlation=0,
                              normalization=False,
                              covariance_design=0,
                              eval_points=9973):
    spacing = end - start
    result = []
    linspace_size = 2000
    x = np.linspace(0, 1, linspace_size)
    if display:
        plt.close('all')
        plt.figure(figsize=(8, 6))
    for l1 in range(start, end - 1):
        for l2 in range(l1 + 1, end):
            values = empirical_law(predictors_number,
                                   sample_size,
                                   start=l1,
                                   end=end,
                                   middle=l2,
                                   iterations=int(iterations),
                                   display=0,
                                   sparsity=int(sparsity),
                                   sigma=sigma,
                                   noise_correlation=noise_correlation,
                                   normalization=normalization,
                                   covariance_design=covariance_design,
                                   eval_points=int(eval_points))
            result += [values, "$\lambda_{%s}$, $\lambda_{%s}$, $\lambda_{%s}$" % (l1, l2, end)]
            if display:
                plt.subplot(spacing - 1, spacing - 1, (l1 - start) * (spacing - 1) + (l2 - start))
                a = np.asarray(values)
                z = np.arange(0, 1, 1. / len(a))
                t = np.sort(a)
                plt.ylim(0, 1)
                plt.xlim(0, 1)
                plt.xticks([0, 0.5, 1], fontsize='x-small')
                plt.yticks([0, 1], fontsize='x-small')
                plt.step(t, z, color="lightcoral", lw=2)
                plt.plot(x, x, color="yellowgreen", linestyle="--")
                plt.legend([r'CDF of $\widehat\alpha_{%s,%s,%s}$' % (l1, l2, end)], fontsize='small', loc=2)

    if display:
        plt.tight_layout()
        plt.savefig('laws.png', format='png', dpi=600)
        plt.show()

    return result


# %%
# % Recursive formulation of the LAR + computation of the correlations rho
def rec(R, Z, T):
    regressed_process = (T < 1) * (Z / (1 - (T != 1) * T))
    val_lambda = np.amax(regressed_process)
    signed_index = np.argmax(regressed_process)
    p = int(np.size(Z) / 2)
    val_index = int(signed_index % p)

    # % safe division
    def safe_div(x, y):
        if y == 0:
            return 0
        return x / y

    # % parameters
    val_rho = safe_div(np.sqrt(R[signed_index, signed_index]),
                       1 - T[signed_index])
    x_return = R[signed_index, ] / R[signed_index, signed_index]
    R_return = R - np.dot(x_return.reshape(np.size(x_return), 1),
                          np.asarray(R[signed_index]).reshape(1, np.size(x_return)))
    # % set to 0 the variance-covariance of the chosen index
    R_return[val_index, ] = np.zeros(2 * p)
    R_return[:, val_index] = np.zeros(2 * p)
    R_return[val_index + p, ] = np.zeros(2 * p)
    R_return[:, val_index + p] = np.zeros(2 * p)
    # % residuals
    Z_return = Z - Z[signed_index] * x_return
    Z_return[val_index] = 0
    Z_return[val_index + p] = 0
    # % parameter theta: regressed coefficient (<1 implies Irrepresentability)
    T_return = T + (1 - T[signed_index]) * x_return
    T_return[val_index] = 0
    T_return[val_index + p] = 0
    return [R_return, Z_return, T_return, val_lambda, val_index, val_rho]


def lar_rec(X, Y,
            kmax=0,
            normalization=False,
            noise_correlation=0):
    sample_size, predictors_number = np.shape(X)

    if normalization:
        X = np.dot(X, np.diag(np.dot(np.ones((1, np.shape(X)[0])), X * X)[0] ** (-0.5)))

    if np.all(noise_correlation) == 0:
        noise_correlation = np.identity(sample_size)

    A = np.dot(np.matrix.transpose(X),
               np.linalg.inv(noise_correlation))

    R_bar = np.dot(A, X)
    Z = np.dot(A, Y)
    Z = np.concatenate([Z, -Z])
    R = np.block(
        [[R_bar, -R_bar],
         [-R_bar, R_bar]
         ])

    if kmax == 0:
        kmax = min(sample_size, predictors_number, np.linalg.matrix_rank(R, hermitian=True))

    T = np.zeros(np.size(Z))
    lambdas = np.zeros(kmax)
    indexes = np.zeros(kmax)
    correls = np.zeros(kmax)
    var_R = R
    var_Z = Z

    for k in range(kmax):
        var_R, var_Z, T, lambdas[k], indexes[k], correls[k] = rec(var_R, var_Z, T)

    return [lambdas, indexes, correls, R, Z]


# %%
# % Compute the Empirical Irrepresentable Check
def get_order(var_Z, var_R):
    T = np.zeros(np.size(var_Z))
    irrepresentable = True
    order_irrep = 0
    while irrepresentable:
        var_R, var_Z, T, __, __, __ = rec(var_R, var_Z, T)
        irrepresentable = bool(np.amax(np.abs(T)) < 1)
        order_irrep += 1
    return order_irrep


# % Compute the Empirical Irrepresentable Check
def get_residual(var_R, var_Z, k):
    T = np.zeros(np.size(var_Z))
    for n in range(k):
        var_R, var_Z, T, __, __, __ = rec(var_R, var_Z, T)
    return [var_R, var_Z]


# % Compute the esitmation of the variance of the residuals
def get_variance(R, Z, t):
    global R2
    R1, Z1 = get_residual(R, Z, t)
    w, v = np.linalg.eig(R1)
    w = np.real(w)

    # We take R2 = R1^(-1/2)
    w2 = np.zeros(np.shape(w))
    for k in range(np.size(w)):
        if abs(w[k]) > 1e-8:
            w2[k] = abs(w[k]) ** (-0.5)
            R2 = np.real(np.dot(v, np.dot(np.diag(w2), np.transpose(v))))
            Z2 = np.dot(R2, Z1)

    # We compute the variance estimation
    d = np.linalg.matrix_rank(R2, hermitian=True)
    V2 = v[:, 0:d]
    Y2 = np.real(np.dot(np.transpose(V2), Z2))
    var = (np.sum(Y2 ** 2) - d * ((np.sum(Y2) / d) ** 2)) / (d - 1)
    return var


# %%
def observed_significance_spacing(lars, sigma, start):
    lambdas, indexes, correls = lars
    middle = start + 1
    end = middle + 1
    if start != 0:
        lambda_a = lambdas[start - 1]
    else:
        lambda_a = np.inf

    lambda_b = lambdas[middle - 1]
    rho_b = correls[middle - 1]
    lambda_c = lambdas[end - 1]
    num2 = (erfc(lambda_b / (np.sqrt(2) * sigma * rho_b)) - erfc(lambda_a / (np.sqrt(2) * sigma * rho_b)))
    den2 = (erfc(lambda_c / (np.sqrt(2) * sigma * rho_b)) - erfc(lambda_a / (np.sqrt(2) * sigma * rho_b)))
    hat_alpha = num2 / den2
    return hat_alpha


# %% MCQMC Fast-CBC method

# % closest prime less than n
def prime(n):
    test = n
    while not isPrime(test):
        test -= 1
    return int(test)


# % prime test
def isPrime(n):
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


# % vector generating the lattice for cubature
def fast_rank_1(n,
                s_max,
                omega=lambda x: x ** 2 - x + 1 / 6,
                gamma=0.9,
                print_CBC=False):
    # % n has to be prime
    if isPrime(n):
        n_prime = n
    else:
        n_prime = int(prime(n))
        if print_CBC:
            print("Warning: number of points is not a prime number, changed to n=%s points" % n_prime)
    # % generating vector
    z = np.zeros(s_max)
    #    e2  = np.zeros(s_max)

    m = int((n_prime - 1) / 2)
    q = np.ones(m)
    q0 = 1

    gamma_vec = np.cumprod(gamma * np.ones(s_max))
    beta_vec = 1 + gamma_vec / 3
    #    cumbeta     = np.cumprod(beta_vec)

    g = np.random.randint(2, n_prime)
    perm = np.zeros(m)
    temp = 1

    for j in range(m):
        perm[j] = int(temp)
        temp = temp * g % n_prime
    perm = np.minimum(perm, n_prime - perm)
    psi = omega(perm / n_prime)
    psi0 = omega(0)
    fft_psi = np.fft.fft(psi)

    for s in range(s_max):
        E2 = np.fft.ifft(fft_psi * np.fft.fft(q))
        E2 = np.real(E2)
        w = np.argmin(E2)
        #        min_E2  = E2[w]
        if s == 0:
            w = 1
        #            noise=np.abs(E2[0]-min_E2)
        z[s] = perm[w]
        #        e2[s]   = -cumbeta[s]+(beta_vec[s]*(q0+2*np.sum(q))+gamma_vec[s]*(psi0*q0+2*min_E2))/n_prime
        ppsi = np.concatenate((psi[w::-1], psi[:w:-1]), axis=0)
        q = (beta_vec[s] + gamma_vec[s] * ppsi) * q
        q0 = (beta_vec[s] + gamma_vec[s] * psi0) * q0
        if print_CBC:
            print("s=%s, z=%s, w=%s" % (s, z[s], w))

    return [z, n_prime]


def observed_significance_CBC(lars,
                              sigma,
                              start,
                              end,
                              middle=-1,
                              eval_points=9973,  # 99991,
                              rep=20):  # rep is used to repeat the computations rep times to estimate the precision
    # error
    # % calculus is different if middle = start+1 (int_the_middle=False in this case)
    lambdas, indexes, correls = lars
    lambdas = lambdas / sigma
    sigma = 1
    restarts = 1
    normal_cut = 4

    in_the_middle = True

    if middle == -1:
        middle = start + 1
    if middle == start + 1:
        in_the_middle = False

    lambda_a = lambdas[int(start - 1)]
    if start == 0:
        lambda_a = np.max([normal_cut * np.log(len(lambdas)) * sigma * correls[0],
                           2 * normal_cut * lambdas[0]])  # should be np.inf but this technique only works on hypercubes
    lambda_b = lambdas[middle - 1]
    lambda_c = lambdas[end - 1]

    variables = []
    for k in range(start, end - 1):
        var = "%s" % k
        variables += [var]
    s_max = len(variables)

    def stat_pdf(*args):
        args_ordered = np.asarray(sorted(args, reverse=True))
        temp = 1
        for k in range(s_max):
            temp *= np.exp((lambda_c ** 2 - (args_ordered[k] ** 2)) / (2 * (sigma * correls[int(variables[k])] ** 2)))
        return temp

    def stat_pdf_with_middle(*args):
        args_ordered = np.asarray(sorted(args, reverse=True))
        temp = 1
        for k in range(s_max):
            if (k == middle - start - 1) & (args_ordered[k] > lambda_b):
                return 0
            temp *= np.exp((lambda_c ** 2 - (args_ordered[k] ** 2)) / (2 * (sigma * correls[int(variables[k])] ** 2)))
        return temp

    alpha = []
    for t in range(restarts):
        [z, n_prime] = fast_rank_1(eval_points, s_max)
        if not in_the_middle:
            for r in range(rep):
                #                z_current = z
                #                if r!=0:
                z_current = z + np.random.uniform(0, 1, size=s_max) % 1

                def stat_cdf(t):
                    values = []
                    for k in range(n_prime):
                        current_point = np.asarray(lambda_c + (t - lambda_c) * (k * z_current / n_prime % 1))
                        values += [stat_pdf(*current_point)]
                    return ((t - lambda_c) ** s_max) * np.mean(np.asarray(values))

                stat1 = stat_cdf(lambda_b)
                stat2 = stat_cdf(lambda_a)
                alpha += [1 - stat1 / stat2]
        else:
            for r in range(rep):
                z_current = z
                if r != 0:
                    z_current = z + np.sqrt(n_prime) * np.random.uniform(-1, 1, size=s_max)
                values = []
                values_normaliszation = []
                for k in range(n_prime):
                    current_point = np.asarray(lambda_c + (lambda_a - lambda_c) * (k * z_current / n_prime % 1))
                    values += [stat_pdf_with_middle(*current_point)]
                    values_normaliszation += [stat_pdf(*current_point)]
                stat1 = ((lambda_a - lambda_c) ** s_max) * np.mean(np.asarray(values))
                stat2 = ((lambda_a - lambda_c) ** s_max) * np.mean(np.asarray(values_normaliszation))
                alpha += [1 - stat1 / stat2]

    return np.median(alpha), 4 * np.std(alpha), np.mean(alpha)


# %%
def stacked_bar(data, series_labels, category_labels=None,
                show_values=False, value_format="{}", y_label=None,
                grid=True, reverse=False):
    """Plots a stacked bar chart with the data and labels provided.

    Keyword arguments:
    data            -- 2-dimensional numpy array or nested list
                       containing data for each series in rows
    series_labels   -- list of series labels (these appear in
                       the legend)
    category_labels -- list of category labels (these appear
                       on the x-axis)
    show_values     -- If True then numeric value labels will
                       be shown on each bar
    value_format    -- Format string for numeric value labels
                       (default is "{}")
    y_label         -- Label for y-axis (str)
    grid            -- If True display grid
    reverse         -- If True reverse the order that the
                       series are displayed (left-to-right
                       or right-to-left)
    """

    ny = len(data[0])
    ind = list(range(ny))

    axes = []
    cum_size = np.zeros(ny)

    data = np.array(data)

    if reverse:
        data = np.flip(data, axis=1)
        category_labels = reversed(category_labels)

    for i, row_data in enumerate(data):
        axes.append(plt.bar(ind, row_data, bottom=cum_size,
                            label=series_labels[i]))
        cum_size += row_data

    if category_labels:
        plt.xticks(ind, category_labels)

    if y_label:
        plt.ylabel(y_label)

    plt.legend()

    if grid:
        plt.grid()

    if show_values:
        for axis in axes:
            for bar in axis:
                w, h = bar.get_width(), bar.get_height()
                plt.text(bar.get_x() + w / 2, bar.get_y() + h / 2,
                         value_format.format(h), ha="center",
                         va="center")
