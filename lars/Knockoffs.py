import numpy as np
from cvxopt import solvers
from cvxopt.solvers import conelp
from sklearn import linear_model
from cvxopt import matrix
import scipy.cluster.hierarchy
from statsmodels.stats.moment_helpers import cov2corr

class Knockoffs():

	def __init__(self):
		pass

	def fdr_power_knockoffs(self, X, y, true_support, alpha=0.1, mode='equicorrelated', knockoff_plus=True, **kwargs):
		support_knockoff = self.support_fdr_knockoffs(X, y, alpha=alpha, mode=mode, knockoff_plus=knockoff_plus, **kwargs)
		FDR = self.FDR(support_knockoff, true_support)
		power = self.power(support_knockoff, true_support)
		return FDR, power

	def support_fdr_knockoffs(self, X, y, alpha=0.1, mode='equicorrelated', knockoff_plus=True, **kwargs):
		self.n, self.p = X.shape
		Xcorr = X / np.tile(np.linalg.norm(X, axis=0).reshape(1,-1),(self.n,1))
		Xcorr[np.where(np.isnan(Xcorr))] = 0
		knockoff = self.knockoff(Xcorr, mode=mode, **kwargs)
		design = np.hstack((Xcorr,knockoff))
		alphas, actives, coefs = linear_model.lars_path(design, y, method='lasso', verbose=False)
		Z = np.zeros(self.p)
		Ztilde = np.zeros(self.p)
		count = 0
		already_seen = np.zeros(2*self.p)
		for index in actives:
			if not already_seen[index]:
				if index < self.p:
					Z[index] = alphas[count]
				else:
					Ztilde[index % self.p] = alphas[count]
				already_seen[index] = 1
				count += 1  
		W = np.zeros(self.p)
		for i in range(self.p):
			W[i] = max(Z[i], Ztilde[i])
			if Z[i]<Ztilde[i]:
				W[i] *= -1
		Wsort = np.sort(np.abs(W))
		T = -1
		stop = False
		while (not(stop) and T<self.p-1):
			T += 1
			plus = np.sum(Wsort[T]<=W)
			moins = np.sum(W<=-Wsort[T])
			if knockoff_plus:
				moins += 1
			stop = (moins/max(1, plus))<=alpha
		return np.where(W >= Wsort[T])[0]

	def knockoff(self, X, mode='equicorrelated', **kwargs):
		Sigma = X.T @ X
		if mode=='equicorrelated':
			s = self.s_equicorrelated(Sigma)
			return self.s2knockoff(X, Sigma, s)
		elif mode=='SDP':
			s = self.s_SDP(Sigma)
			return self.s2knockoff(X, Sigma, s)
		elif mode=='ASDP':
			s = self.s_ASDP(Sigma)
			return self.s2knockoff(X, Sigma, s, **kwargs)

	def s2knockoff(self, X, Sigma, s):
		invSigma = np.linalg.inv(Sigma)
		A = 2*np.diag(s) - np.diag(s) @ invSigma @ np.diag(s)
		w, v = np.linalg.eig(A)
		w = np.real(w)
		w *= (w>0)
		C = np.diag(np.sqrt(w)) @ v.T
		u, __, __ = np.linalg.svd(X)
		u = u[:,:self.p]
		proj = np.eye(self.n) - u @ np.linalg.pinv(u)
		U, __, __ = np.linalg.svd(proj)
		U = U[:,:self.p]
		return (X @ (np.eye(self.p) - invSigma @ np.diag(s)) + U @ C)

	def s_equicorrelated(self, Sigma):
		lambda_min = np.min(np.linalg.eigvals(Sigma))
		s = min(2*lambda_min, 1) * np.ones(self.p)
		s *= s>0
		# Compensate for numerical errors (feasibility)
		# psd = False
		# s_eps = 1e-8
		# while not psd:
		# 	psd = np.all(np.linalg.eigvals(2*Sigma-diag(s*(1-s_eps)))> 0)
		# 	if not psd:
		# 		s_eps = s_eps*10
		# s = s*(1-s_eps)
		return s

	def s_SDP(self, Sigma):
		p = Sigma.shape[0]
		c = -np.ones(p)
		c = matrix(c)
		G = np.zeros((2*p+p**2,p))
		G[:p,:] = np.eye(p)
		G[p:2*p,:] = -np.eye(p)
		for i in range(p):
			G[2*p+p*i,i] = 1
		G = matrix(G)
		h = np.ones(2*p+p**2)
		h[p:2*p] *= 0
		h[2*p:] *= 2*(Sigma).reshape(-1)
		h = matrix(h)
		dims = {'l': 2*p, 'q': [], 's': [p]}
		solvers.options['show_progress'] = False
		sol = conelp(c, G, h, dims)
		s = np.array(sol['x']).reshape(-1)
		return s

	def s_ASDP(self, Sigma, **kwargs):
		""" Section 3.4.2 : Panning for Gold:Model-X Knockoffs for High-dimensional Controlled Variable Selection """
		maxclustersize = kwargs.get('maxclustersize', self.p)
		accuracy = kwargs.get('accuracy', 1e-2)
		max_iter = kwargs.get('max_iter', 100)
		linkage = scipy.cluster.hierarchy.linkage(Sigma, method='single', metric='euclidean')
		groups = {i:[i] for i in range(self.p)}
		next_group = self.p 
		for i in range(linkage.shape[0]):
			try:
				group1 = groups[linkage[i,0]]
				group2 = groups[linkage[i,1]]
				if len(group1)+len(group2) <= maxclustersize:
					groups[next_group] = group1 + group2
					del groups[linkage[i,0]]
					del groups[linkage[i,1]]
			except:
				pass
			next_group += 1

		blocks = list(groups.values())
		s = np.zeros(self.p)
		for block in blocks:
			temp = Sigma[block,:]
			shat = self.s_SDP(temp[:,block])
			s[block] = shat
		# Gershgorin circle theorem
		maxgamma = min(1, np.min(2*np.diag(Sigma)/s))
		mingamma = 0
		nbite = 0
		while (nbite<max_iter and np.abs(maxgamma-mingamma)<accuracy):
			gamma = (maxgamma + mingamma)/2
			nbite += 1
			try:
				__ = np.linalg.cholesky(2*Sigma - np.diag(gamma*s))
				mingamma = gamma
			except:
				maxgamma = gamma
		gamma = (maxgamma + mingamma)/2
		return gamma*s