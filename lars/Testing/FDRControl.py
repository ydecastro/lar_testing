import numpy as np
import matplotlib.pyplot as plt

class FDRControl:
	def __init__(self):
		pass

	def compute_pvalues(self, number_hyp=None, sigma=None, K1=None, K2=None):
		if number_hyp is None:
			number_hyp = self.get_order()
		hat_alpha = np.zeros(number_hyp)
		if sigma is None:
			sigma = self.get_std(K1=K1, K2=K2)
		for start in range(number_hyp):
			hat_alpha[start] = self.observed_significance_spacing(start, sigma=sigma)
		return hat_alpha

	def display_pvalues(self, alpha, number_hyp=None, sigma=None, K1=None, K2=None):
		pvalues = self.compute_pvalues(number_hyp=number_hyp, sigma=sigma, K1=K1, K2=K2)
		number_hyp = len(pvalues)
		p_ordered = sorted(pvalues, reverse=False)
		x = np.cumsum(np.ones(len(pvalues)))
		plt.plot(x, p_ordered)
		plt.plot(x, alpha*x/number_hyp)
		plt.tight_layout()
		plt.xlabel(r'$k$')
		plt.xlim([1,24])
		plt.ylim([0,0.2])
		plt.ylabel(r'$p_{(k)}$')
		plt.title('Ordered p-Values for Multiple Spacings')
		plt.show()

	def support_fdr_lars(self, X, y, alpha=0.1, kmax=0, normalization=True, lars_algorithm='recursive', number_hyp=None, sigma=None, K1=None, K2=None):
		#self.compute_lars_path(X, y, kmax=kmax, normalization=normalization, lars_algorithm=lars_algorithm)
		return self.fdr_control(alpha, number_hyp=number_hyp, sigma=sigma, K1=K1, K2=K2)

	def fdr_control(self, alpha, number_hyp=None, sigma=None, K1=None, K2=None):
		hat_alpha = self.compute_pvalues(number_hyp=number_hyp, sigma=sigma, K1=K1, K2=K2)
		number_hyp = len(hat_alpha)

		# The rejected hyptoheses are given by:
		hat_alpha[np.isnan(hat_alpha)] = 0
		p_ordered = sorted(hat_alpha, reverse=False)
		k = number_hyp-1
		condition_not_seen = True
		while (p_ordered[k]>alpha*(k+1)/number_hyp and k-1>=0):
			k -= 1
		if k==0 and p_ordered[k]>alpha*(k+1)/number_hyp:
			return []
		else:	
			rejected = hat_alpha<=alpha*k/number_hyp

			# whose indexes are: 
			rej = 1*rejected
			ind = self.larspath['indexes'][:number_hyp]*rej
			ind = ind.astype(int)
			ind = ind[ind!=0]
			return ind

	def fdr_power_lars(self, X, y, true_support, alpha, number_hyp=None,  kmax=0, normalization=True, lars_algorithm='recursive', sigma=None, K1=None, K2=None):
		self.compute_lars_path(X, y, kmax=kmax, normalization=normalization, lars_algorithm=lars_algorithm)
		support = self.fdr_control(alpha, number_hyp=number_hyp, sigma=sigma, K1=K1, K2=K2)
		FDR = self.FDR(support, true_support)
		power = self.power(support, true_support)
		return FDR, power

	# def fdr_power_lars(self, true_support, alpha, number_hyp=None, sigma=None, K1=None, K2=None):
	# 	support = self.fdr_control(alpha, number_hyp=number_hyp, sigma=sigma, K1=K1, K2=K2)
	# 	FDR = self.FDR(support, true_support)
	# 	power = self.power(support, true_support)
	# 	return FDR, power

	def stacked_bar(data, series_labels, category_labels=None,
					show_values=False, value_format="{}", y_label=None,
					grid=True, reverse=False):
		"""Plots a stacked bar chart with the data and labels provided.

		Keyword arguments:
		data			-- 2-dimensional numpy array or nested list
						   containing data for each series in rows
		series_labels   -- list of series labels (these appear in
						   the legend)
		category_labels -- list of category labels (these appear
						   on the x-axis)
		show_values	 -- If True then numeric value labels will 
						   be shown on each bar
		value_format	-- Format string for numeric value labels
						   (default is "{}")
		y_label		 -- Label for y-axis (str)
		grid			-- If True display grid
		reverse		 -- If True reverse the order that the
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