from scipy.special import erfc
from  .IntegrationMethods.CurbatureByLatticeRule import *
import numpy as np


class IntegrationMethods():
	def __init__(self):
		pass

	def integrate(self, f, low, up, t, s_max, integration_method='qmc_curbature_by_lattice', **kwargs):
		if integration_method == 'qmc_curbature_by_lattice':
			return self.qmc_curbature_by_lattice(f, low, up, t, s_max, **kwargs)

	def qmc_curbature_by_lattice(self, f, low, up, t, s_max, **kwargs):
		eval_points = kwargs.get('eval_points', 9973)
		[z, n_prime] = fast_rank_1(eval_points, s_max)
		z_current = z + np.random.uniform(0, 1, size=s_max) % 1
		values = []
		for k in range(n_prime):
			current_point = np.asarray(low + (t - low) * (k * z_current / n_prime % 1))
			values += [f(*current_point)]
		return ((t - low) ** s_max) * np.mean(np.asarray(values))



class pValues(IntegrationMethods):
	def __init__(self, ):
		super(IntegrationMethods, self).__init__()

	def observed_significance_spacing(self, start, sigma=None):
		if sigma is None:
			if self.sigma is not None:
				sigma = self.sigma
			else:
				sigma = self.get_std()
		middle = start + 1
		end = middle + 1
		if start != 0:
			lambda_a = self.larspath['lambdas'][start - 1]
		else:
			lambda_a = np.inf

		lambda_b = self.larspath['lambdas'][middle - 1]
		rho_b = self.larspath['correls'][middle - 1]
		lambda_c = self.larspath['lambdas'][end - 1]
		num2 = (erfc(lambda_b / (np.sqrt(2) * sigma * rho_b)) - erfc(lambda_a / (np.sqrt(2) * sigma * rho_b)))
		den2 = (erfc(lambda_c / (np.sqrt(2) * sigma * rho_b)) - erfc(lambda_a / (np.sqrt(2) * sigma * rho_b)))

		# % safe division
		def safe_div(x, y):
			if y == 0:
				return 0
			return x / y

		hat_alpha = safe_div(num2, den2)
		return hat_alpha


	def observed_significance(self,
							sigma,
							start,
							end,
							middle=-1,
							**kwargs):

		lambdas = np.copy(self.larspath['lambdas']) / sigma
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
			# lambda_c ** 2 iis used in the pdf only to improve the accuracy of computation (basically to avoid to visit tails of the exponential)
			args_ordered = np.asarray(sorted(args, reverse=True))
			temp = 1
			for k in range(s_max):
				temp *= np.exp((lambda_c ** 2 - (args_ordered[k] ** 2)) / (2 * (sigma * self.larspath['correls'][int(variables[k])] ** 2)))
			return temp

		def stat_pdf_with_middle(*args):
			args_ordered = np.asarray(sorted(args, reverse=True))
			temp = 1
			for k in range(s_max):
				if (k == middle - start - 1) & (args_ordered[k] > lambda_b):
					return 0
				temp *= np.exp((lambda_c ** 2 - (args_ordered[k] ** 2)) / (2 * (sigma * self.larspath['correls'][int(variables[k])] ** 2)))
			return temp

		alpha = []
		for ite in range(restarts):
			if not in_the_middle:
				stat1 = self.integrate(stat_pdf, lambda_c, lambda_a, lambda_b, s_max, **kwargs)
				stat2 = self.integrate(stat_pdf, lambda_c, lambda_a, lambda_a, s_max, **kwargs)
			else:
				stat1 = self.integrate(stat_pdf_with_middle, lambda_c, lambda_a, lambda_a, s_max, **kwargs)
				stat2 = self.integrate(stat_pdf, lambda_c, lambda_a, lambda_a, s_max, **kwargs)
			alpha += [1 - stat1 / stat2]

		return np.median(alpha), 4 * np.std(alpha), np.mean(alpha)
