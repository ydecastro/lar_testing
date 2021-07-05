import numpy as np

class StoppingTime():
	def __init__(self):
		pass

	def select_model_size(self, hat_alpha, alpha, stoppingtimecriterion='consecutive_accepts', **kwargs):
		if stoppingtimecriterion == 'consecutive_accepts':
			return self.consecutive_accepts(hat_alpha, alpha, **kwargs)

	def consecutive_accepts(self, hat_alpha, alpha, **kwargs):
		num_accepts = kwargs.get('num_accepts', 3)
		display = kwargs.get('display', False)
		reject = 1*(hat_alpha<alpha)
		test = np.zeros(len(reject)-num_accepts+1)
		for k in range(num_accepts):
			test += reject[k:len(reject)-(num_accepts-k-1)]

		if display:
			plt.figure(1)
			plt.plot(test)
			plt.title('Number of rejects on the next '+str(num_accepts)+' tests')

		candidates = np.where(test == 0)
		assert len(candidates[0])>0, "No {0} consecutive accepts were found in the lars path for the stopping time.".format(num_accepts)
		hat_m = int(candidates[0][0])+num_accepts
		return hat_m

	# % Compute the Empirical Irrepresentable Check
	def get_order(self):
		try:
			return self.order_irrep
		except:
			var_Z, var_R = self.Z, self.R
			T = np.zeros(np.size(var_Z))
			irrepresentable = True
			order_irrep = 0
			while irrepresentable:
				var_R, var_Z, T, __, __, __, __ = self.rec(var_R, var_Z, T)
				irrepresentable = bool(np.amax(np.abs(T)) < 1)
				order_irrep += 1
			self.order_irrep = order_irrep
			return order_irrep