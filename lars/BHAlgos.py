from scipy import stats
import numpy as np


def hinge_exp(t, C=2):
	if t <= 1-1/C:
		return 0
	else:
		return C * np.log(1/(C*(1-t)))


def seqstep(t, C=2):
	if t <= 1-1/C:
		return 0
	else:
		return C

def forward_stop(t, C=None):
	return np.log(1/(1-t))
		

class BHAlgos():
	def __init__(self):
		pass

	def compute_bh_pvalues(self, X, y):
		n,p = np.shape(X)
		invcov = np.linalg.inv(X.T @ X)
		beta = invcov @ X.T @ y
		sse = np.sum( (y- X@beta)**2) / (n-p)
		p_values = []
		for i in range(p):
			p_values.append( 2*(1-stats.t.cdf(np.abs(beta[i])/np.sqrt(invcov[i,i]*sse),n-p)))
		return np.array(p_values)

	def fdr_power_storey_bh(self, X, y, true_support, lamb=0.5, alpha=0.2, p_values=None):
		support_knockoff = self.support_fdr_storey_bh(X, y, lamb=lamb, alpha=alpha, p_values=p_values)
		FDR = self.FDR(support_knockoff, true_support)
		power = self.power(support_knockoff, true_support)
		return FDR, power

	def support_fdr_storey_bh(self, X, y, lamb=0.5, alpha=0.2, p_values=None):
		p_values = self.compute_bh_pvalues(X,y)
		sorted_ind = np.argsort(p_values).astype(int)
		p_ordered = p_values[sorted_ind]
		
		number_hyp = len(p_values)
		r_lambda = np.sum(p_ordered < lamb)
		w_lambda = number_hyp - r_lambda
		# Estimation of the overall proportion of true nulls
		hatpi0 = w_lambda / (1-lamb)
		fdrs = (hatpi0 * p_ordered) / np.cumsum(np.ones(number_hyp))
		k = number_hyp-1
		while (fdrs[k]>alpha and k-1>=0):
			k -= 1
		if k==0 and fdrs[k]>alpha:
			return []
		else:
			return np.where(p_values <= alpha*(k+1)/hatpi0)[0]

	def fdr_power_bh(self, X, y, true_support, alpha=0.2, p_values=None):
		support_knockoff = self.support_fdr_bh(X, y, alpha=alpha, p_values=p_values)
		FDR = self.FDR(support_knockoff, true_support)
		power = self.power(support_knockoff, true_support)
		return FDR, power

	def support_fdr_bh(self, X, y, alpha=0.2, p_values=None):
		return self.support_fdr_storey_bh(X, y, lamb=0, alpha=alpha, p_values=p_values)

	def fdr_power_forward_stop(self, X, y, true_support, alpha=0.2):
		support_forward_stop = self.support_fdr_forward_stop(X, y, alpha=alpha)
		FDR = self.FDR(support_forward_stop, true_support)
		power = self.power(support_forward_stop, true_support)
		return FDR, power

	def support_fdr_forward_stop(self, X, y, alpha=0.2):
		"Forward Stop Procedure from G’Sell et al."
		p_values = self.compute_bh_pvalues(X,y)
		sorted_ind = np.argsort(p_values).astype(int)
		p_ordered = p_values[sorted_ind]
		number_hyp = len(p_values)
		k = -1
		cumsum = np.cumsum(np.log(1/(1-p_ordered)))
		while (cumsum[k+1]<=alpha*(k+2) and k+2<number_hyp):
			k += 1
		return sorted_ind[:k+1]

	def fdr_power_seqstep(self, X, y, true_support, alpha=0.2, seqstep_plus=True):
		support_seqstep = self.support_fdr_seqstep(X, y, alpha=alpha, seqstep_plus=seqstep_plus)
		FDR = self.FDR(support_seqstep, true_support)
		power = self.power(support_seqstep, true_support)
		return FDR, power

	def support_fdr_seqstep(self, X, y, C=2, alpha=0.2, seqstep_plus=True):
		assert C>1, "In the Sequential Step-up Procedure (SeqStep) of Barber and Candès, the hyperparameter C needs to be chosen strictly larger than 1."

	def support_fdr_accumulation(self, X, y, accumulation_func='hinge_exp', C=1, alpha=0.2, plus=True, p_values=None):
		
		if p_values is None:
			p_values = self.compute_pvalues(number_hyp=X.shape[1]-50, K1=X.shape[1]-50)
		sorted_ind = self.larspath['indexes'].astype(int)[:len(p_values)]
		number_hyp = len(p_values)

		k = number_hyp-1
		cumsum = np.cumsum([globals()[accumulation_func](p, C=C) for p in p_values])
		if plus:
			cumsum = C + cumsum
			while (cumsum[k]>alpha*(k+2) and k-1>=0):
				k -= 1
			if k==0 and cumsum[k]>alpha*(k+2):
				return []
			else:
				return sorted_ind[:k+1]
		else:
			while (cumsum[k]>alpha*(k+1) and k-1>=0):
				k -= 1
			if k==0 and cumsum[k]>alpha*(k+1):
				return []
			else:
				return sorted_ind[:k+1]
		# k = -1
		# condition_not_seen = True
		# cumsum = np.cumsum([globals()[accumulation_func](p, C=C) for p in p_ordered])
		# if plus:
		# 	cumsum = C + cumsum
		# 	while (cumsum[k+1]<=alpha*(k+3) or condition_not_seen) and k+2<number_hyp:
		# 		if cumsum[k+1]<=alpha*(k+3):
		# 			condition_not_seen = False
		# 		k += 1
		# else:
		# 	while (cumsum[k+1]<=alpha*(k+2) or condition_not_seen) and k+2<number_hyp:
		# 		if cumsum[k+1]<=alpha*(k+2):
		# 			condition_not_seen = False
		# 		k += 1
		# if condition_not_seen:
		# 	return []
		# else:
		# 	return sorted_ind[:k+1]

	def fdr_power_accumulation(self, X, y, true_support, alpha=0.2, accumulation_func='hinge_exp', plus=True, p_values
		=None):
		support = self.support_fdr_hinge_exp(X, y, alpha=alpha, accumulation_func=accumulation_func, plus=plus, p_values=p_values)
		FDR = self.FDR(support, true_support)
		power = self.power(support, true_support)
		return FDR, power

	# def support_sabha(self, X, y, tau=0.5, epsilon=0.1, alpha=0.2, mode='monotone'):
	# 	p_values = self.compute_pvalues(X,y)
	# 	sorted_ind = np.argsort(p_values).astype(int)
	# 	p_ordered = p_values[sorted_ind]
	# 	number_hyp = len(p_values)
	# 	if mode == 'monotone':

	# 	else:	
	# 		p_ordered = epsilon * (p_ordered<epsilon) + (p_ordered>=epsilon)
	# 	return 


