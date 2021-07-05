###################################
# The part of the code computes the matrix M used for the Debiased lasso
###################################
import numpy as np
import scipy as sc
from sklearn import linear_model
import scipy as sc
import cvxpy as cp
import matplotlib.pyplot as plt
import random

def softThreshold(x,lamb):
    #
    # Standard soft thresholding
    #
    if (x>lamb):
       return (x-lamb)
    else:
        if (x< (-lamb)):
            return (x+lamb)
        else:
            return (0)

def inverseLinftyOneRow( sigma, i, mu, maxiter=50, threshold=1e-2 ):
    p = sigma.shape[0]
    sig = sigma[i,:]
    sig = np.delete(sig, i)
    rho = max(abs(sig)) / sigma[i,i]
    mu0 = rho/(1+rho)
    beta = np.zeros(p)

    if (mu >= mu0):
        beta[i] = (1-mu0)/sigma[i,i]
        return (beta, 0)

    diffnorm2 = 1
    lastnorm2 = 1
    iter = 1
    iterold = 1
    beta[i] = (1-mu0)/sigma[i,i]
    betaold = beta
    sigmatilde = np.copy(sigma)
    for k in range(p):
        sigmatilde[k,k] = 0
    vs = -sigmatilde @ beta

    while ((iter <= maxiter) and (diffnorm2 >= threshold*lastnorm2)):

        for j in range(p):
            oldval = beta[j]
            v = vs[j]
            if (j==i):
                v = v+1
            beta[j] = softThreshold(v,mu)/sigma[j,j]
            if (oldval != beta[j]):
                vs = vs + (oldval-beta[j])*sigmatilde[:,j]

        iter = iter + 1
        if (iter==2*iterold):
            d = beta - betaold
            diffnorm2 = np.linalg.norm(d)
            lastnorm2 = np.linalg.norm(beta)
            iterold = iter
            betaold = np.copy(beta)
            if (iter>10):
                vs = -sigmatilde @ beta

    return (beta, iter)

import math
def inverseLinfty(sigma, n, resol=1.5, mu=None, maxiter=50, threshold=1e-2, verbose = False):
    isgiven = 1
    if (mu is None):
  	    isgiven = 0

    p = sigma.shape[0]
    M = np.zeros((p,p))
    xperc = 0
    xp = round(p/10)
    for i in range(p):
        if ((i % xp)==0):
            xperc = xperc+10
            if (verbose):
                print(xperc)
        if (isgiven==0):
            mu = (1/np.sqrt(n)) * sc.stats.norm.ppf(1-(0.1/(p**2)))

        mustop = 0
        tryno = 1
        incr = 0

        beta = np.zeros(p)
        while ((mustop != 1) and (tryno<10)):
            lastbeta = np.copy(beta)
            beta, iter = inverseLinftyOneRow(sigma, i, mu, maxiter=maxiter, threshold=threshold)
            assert(not(math.isnan(beta[0])))
            if (isgiven==1):
                mustop = 1
            else:
                if (tryno==1):
                    if (iter == (maxiter+1)):
                        incr = 1
                        mu = mu*resol
                    else:
                        incr = 0
                        mu = mu/resol

                if (tryno > 1):
                    if ((incr == 1) and (iter == (maxiter+1))):
                        mu = mu*resol
                    if ((incr == 1) and (iter < (maxiter+1))):
                        mustop = 1;
                    if ((incr == 0) and (iter < (maxiter+1))):
                        mu = mu/resol
                    if ((incr == 0) and (iter == (maxiter+1))):
                        mu = mu*resol
                        beta = lastbeta
                        mustop = 1
            tryno = tryno+1
        M[i,:] = np.copy(beta)
    return (M)

###

def sigma_square_root_lasso(X, y, lamb, nbite=10):
    '''
    Estimate the noise level using the square root lasso
    We alternatively optimize over sigma and theta
    See slide 26/40 in : http://josephsalmon.eu/talks/bounds_slides_CIRM2020.pdf
    '''
    n = np.shape(X)[0]
    sigma = 1
    for i in range(nbite):
        clf = linear_model.Lasso(alpha=sigma * lamb, fit_intercept=False)
        clf.fit(X, y)
        theta_lasso = clf.coef_
        sigma = np.linalg.norm( y - X @ theta_lasso ) / np.sqrt(n)
    return sigma


def FCD(n, p, X, y, level=0.1, sigma=None):
    '''
    False Discovery Rate Control via Debiased Lasso
    '''
    from sklearn import linear_model

    lamb = 2 * np.sqrt(2 * np.log(p)/n) #np.sqrt(sc.stats.norm.ppf(1-(0.1/p))/n)
    clf = linear_model.Lasso(alpha=lamb, fit_intercept=False)
    clf.fit(X, y)
    theta_lasso = clf.coef_

    Sigma_hat = X.T @ X / n
    q = level
    M = inverseLinfty(Sigma_hat, n)
    theta_debias = theta_lasso + M @ X.T @ (y- X @ theta_lasso) / n


    # Compute the test statistic
    Lambda = M @ Sigma_hat @ M.T

    if sigma is None:
        # Estimate the noise using suare-root lasso
        sigma = sigma_square_root_lasso(X, y, lamb)
    else:
        sigma = 1

    # Statistic of test
    statFCD = np.sqrt(n) * theta_debias / (sigma * np.sqrt(np.diag(Lambda)))

    # Compute the threshold t0 (for the rejection region)
    tp = np.sqrt( 2*np.log(p) - 2*np.log(np.log(p)))
    orderedstatFCD = np.sort(np.abs(statFCD))[::-1]
    Rt = 0
    if orderedstatFCD[0]<tp:
        Rt = 0
    else:
        while orderedstatFCD[Rt]>=tp and Rt<p:
            Rt += 1
    if 2*p*(1-sc.stats.norm.cdf(tp))>q*max(1,Rt):
        t0 = np.sqrt(2*np.log(p))
    else:
        while Rt<p and (2*p*(1-sc.stats.norm.cdf(tp))<=q*max(1,Rt)):
            tp = orderedstatFCD[Rt]
            Rt += 1
        if Rt!=p or ( (2*p*(1-sc.stats.norm.cdf(tp))>q*max(1,Rt)) ):
            Rt -= 1
            t0 = sc.stats.norm.ppf(1- (q/(2*p)) * max(1,Rt))
            t0 = max(0,t0)
        else:
            t0 = sc.stats.norm.ppf(1- (q/(2*p)) * max(1,Rt))
            t0 = max(0,t0)

    # Set the estimated support of theta
    Shat = np.where(np.abs(statFCD)>=t0)[0]
    signhat = np.sign(statFCD)
    return signhat, Shat
