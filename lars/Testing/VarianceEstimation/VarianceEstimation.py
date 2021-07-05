import numpy as np

class VarianceEstimation():

	def __init__(self):
		pass

	# % Compute the estimation of the variance of the residuals
	def get_std(self, K1=None, K2=None):
		if K1 is None:
			K1 = self.get_order() + 1
		if K2 is None:
			K2 = self.p
		assert K1<K2, "You need to choose K1<K2"
		# K1 = irrep + (K2-irrep)//2
		global R2
		R1, Z1 = self.get_residual(K1)
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
		return np.sqrt(var)