import matplotlib.pyplot as plt
import numpy as np
from .StoppingTime import StoppingTime
from .VarianceEstimation import VarianceEstimation

class FalseNegativeTest(StoppingTime, VarianceEstimation):
	def __init__(self):
		super(StoppingTime).__init__()
		super(VarianceEstimation).__init__()

	def select_support(self, alpha, start, middle, end, sigma_select):
		number_hyp = self.get_order()
		hat_alpha = np.zeros(number_hyp)
		for step in range(number_hyp):
			hat_alpha[step] = self.observed_significance_spacing(step, sigma=sigma_select)
		hat_alpha[np.isnan(hat_alpha)] = 0 #very small values returns NAN (due a division) and need to be set to 0.
		if start is None:
			start = self.select_model_size(hat_alpha, alpha)
			middle = start+1
		if end is None:
			end = self.get_order()
		return start-1, middle-1, end

	def false_negative_test(self,
							alpha,
							sigma=None,
							start=None, 
							end=None, 
							middle=None,
							K1=None,
							K2=None,
							**kwargs):

		if sigma is not None:
			sigma_select, sigma_test = sigma, sigma
		else:
			if K1 or K2 is None:
				irrep = self.get_order()
				K1 = irrep + (self.p-irrep)//3
				K2 = irrep + 2*((self.p-irrep)//3)
			sigma_select, sigma_test = self.get_std(K1,K2), self.get_std(K2+1,self.p)

		start, middle, end = self.select_support(alpha, start, middle, end, sigma_select)

		pvalues = self.observed_significance(sigma_test,
											start, 
											end,
											middle=middle,
											**kwargs)
		return pvalues[0]



