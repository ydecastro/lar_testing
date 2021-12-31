import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm, binom

from .Testing import FDRControl, FalseNegativeTest
from .pValues import pValues
from .LarsPath import LarsPath

class LARS(FDRControl, FalseNegativeTest, LarsPath, pValues):
	def __init__(self, noise_correlation=0):
		self.noise_correlation = noise_correlation
		self.sigma = None
		super(pValues, self).__init__()
		super(LarsPath, self).__init__()
		super(FDRControl, self).__init__()
		super(FalseNegativeTest, self).__init__()


	# % Generate data
	def generate_data(self,
					predictors_number=100,
					sample_size=100,
					sparsity=0,
					sigma=1,
					noise_correlation=0,
					normalization=False,
					covariance_design=0):
		"""Generate a linear model
		"""
		""" design
		"""
		self.p = predictors_number
		self.n = sample_size
		self.sigma = sigma
		self.sparsity = sparsity
		self.noise_correlation = noise_correlation
		if np.all(covariance_design) == 0:
			X = norm.rvs(0, 1 / np.sqrt(self.n), [self.n, self.p])
		else:
			for k in range(0, self.n):
				X[k, ] = np.random.multivariate_normal(np.zeros(self.p), covariance_design)
		if normalization:
			X = np.dot(X, np.diag(np.dot(np.ones((1, np.shape(X)[0])), X * X)[0] ** (-0.5)))

		""" target vector
		"""
		signs_beta_0 = 2 * binom.rvs(1, 0.5, size=self.sparsity) - 1  # % signs of sparse target vector
		beta_0 = np.concatenate([signs_beta_0, np.zeros(self.p - self.sparsity)])  # % sparse vector

		""" noise vector
		"""
		if np.all(noise_correlation) == 0:
			noise_correlation = np.identity(self.n)
		epsilon = np.random.multivariate_normal(np.zeros(self.n),
												(self.sigma ** 2) * noise_correlation)  # % noise vector

		""" observation
		"""
		Y = np.dot(X, beta_0) + epsilon

		return X, Y, beta_0

	def FDR(self, selected_support, true_support):
		return len(set(selected_support)-set(true_support))/max(len(selected_support),1)

	def power(self, selected_support, true_support):
		return (1-len(set(true_support)-set(selected_support))/max(len(true_support),1))

	def fdr_power(self, true_support, X=None, y=None, alpha=0.1, method='lars', **kwargs):
		if method=='lars':
			sigma = kwargs.get('sigma', None)
			K1 = kwargs.get('K1', None)
			K2 = kwargs.get('K2', None)			
			return self.fdr_power_lars(true_support, alpha, sigma=sigma, K1=K1, K2=K2)
		elif method=='KSDP':
			knockoff_plus = kwargs.get('knockoff_plus', True)
			return self.fdr_power_knockoffs(X, y, true_support, q=alpha, mode='SDP', knockoff_plus=knockoff_plus, **kwargs)
		elif method=='KEQUI':
			knockoff_plus = kwargs.get('knockoff_plus', True)
			return self.fdr_power_knockoffs(X, y, true_support, q=alpha, mode='equicorrelated', knockoff_plus=knockoff_plus, **kwargs)

	def in_out_support(self, true_vars, support2vars, X=None, y=None, alpha=0.1, method='lars', **kwargs):
		if method=='lars':
			sigma = kwargs.get('sigma', None)
			K1 = kwargs.get('K1', None)
			K2 = kwargs.get('K2', None)
			support = self.fdr_control(alpha, sigma=sigma, K1=K1, K2=K2)	
			variables = support2vars[support]
			OUT = len(set(variables)-set(true_vars))
			IN = len(true_vars) - len(set(true_vars)-set(variables))
			return IN, OUT
		
		elif method=='KSDP':
			knockoff_plus = kwargs.get('knockoff_plus', True)
			support = self.support_fdr_knockoffs(X, y, alpha=alpha, mode='SDP', knockoff_plus=knockoff_plus, **kwargs)
			variables = support2vars[support]
			OUT = len(set(variables)-set(true_vars))
			IN = len(true_vars) - len(set(true_vars)-set(variables))
			return IN, OUT

		elif method=='KEQUI':
			knockoff_plus = kwargs.get('knockoff_plus', True)
			support = self.support_fdr_knockoffs(X, y, alpha=alpha, mode='equicorrelated', knockoff_plus=knockoff_plus, **kwargs)
			variables = support2vars[support]
			OUT = len(set(variables)-set(true_vars))
			IN = len(true_vars) - len(set(true_vars)-set(variables)) 
			return IN, OUT