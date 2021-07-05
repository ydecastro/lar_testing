import numpy as np
import matplotlib.pyplot as plt


# % safe division
def safe_div(x, y):
	if y == 0:
		return 0
	return x / y


class LarsPath():
	def __init__(self):
		pass

	# % Compute covariances and correlations
	def compute_covariances_correlations(self, X, Y):
		""" whithening the observation
		"""
		if np.all(self.noise_correlation) == 0:
			noise_correlation = np.identity(self.n)
		A = np.dot(np.matrix.transpose(X),
				   np.linalg.inv(noise_correlation))
		R_bar = np.dot(A, X)
		Z = np.dot(A, Y)
		Z = np.concatenate([Z, -Z])
		R = np.block(
			[[R_bar, -R_bar],
			 [-R_bar, R_bar]
			 ])
		self.Z, self.R = Z, R

	def compute_lars_path(self, X, Y, kmax=0, normalization=False, lars_algorithm='recursive'):
		self.n, self.p = X.shape
		if normalization:
			X = np.dot(X, np.diag(np.dot(np.ones((1, np.shape(X)[0])), X * X)[0] ** (-0.5)))
		self.compute_covariances_correlations(X, Y)

		if lars_algorithm == 'recursive':
			self.__lars_rec(kmax=kmax)
		elif lars_algorithm == 'projection':
			self.lars_projection(kmax=kmax)
		else:
			assert False, "The Lars algorithm specified does not match any implemented algorithm."

	# % Compute the Empirical Irrepresentable Check
	def get_residual(self, k, var_R=None, var_Z=None):
		if (var_R is None) or (var_Z is None):
			var_R = np.copy(self.R)
			var_Z = np.copy(self.Z)
		T = np.zeros(np.size(var_Z))
		for ite in range(k):
			var_R, var_Z, T, __, __, __, __ = self.rec(var_R, var_Z, T)
		return [var_R, var_Z]

	def display_lars_knots(self):
		plt.plot(self.larspath['lambdas'])
		plt.legend([r'recursive'])
		plt.tight_layout()
		plt.xlabel(r'$k$')
		plt.ylabel(r'$\lambda_k$')
		plt.title('Evolution of knots along the LARS path')
		plt.show()

	# % Recursive formulation of the LAR + computation of the correlations rho
	def rec(self, R, Z, T):
		regressed_process = (T < 1) * (Z / (1 - (T != 1) * T))
		val_lambda = np.amax(regressed_process)
		signed_index = np.argmax(regressed_process)
		sign = 2*(signed_index < self.p)-1
		val_index = int(signed_index % self.p)

		# % parameters
		val_rho = safe_div(np.sqrt(R[signed_index, signed_index]),
						   1 - T[signed_index])
		x_return = R[signed_index, ] / R[signed_index, signed_index]
		R_return = R - np.dot(x_return.reshape(np.size(x_return), 1),
							  np.asarray(R[signed_index]).reshape(1, np.size(x_return)))
		# % set to 0 the variance-covariance of the chosen index
		R_return[val_index, ] = np.zeros(2 * self.p)
		R_return[:, val_index] = np.zeros(2 * self.p)
		R_return[val_index + self.p, ] = np.zeros(2 * self.p)
		R_return[:, val_index + self.p] = np.zeros(2 * self.p)
		# % residuals
		Z_return = Z - Z[signed_index] * x_return
		Z_return[val_index] = 0
		Z_return[val_index + self.p] = 0
		# % parameter theta: regressed coefficient (<1 implies Irrepresentability)
		T_return = T + (1 - T[signed_index]) * x_return
		T_return[val_index] = 0
		T_return[val_index + self.p] = 0
		return [R_return, Z_return, T_return, val_lambda, val_index, val_rho, sign]


	def __lars_rec(self, kmax=0):
		if kmax == 0:
			kmax = min(self.n, self.p, np.linalg.matrix_rank(self.R, hermitian=True))

		T = np.zeros(np.size(self.Z))
		lambdas = np.zeros(kmax)
		indexes = np.zeros(kmax)
		correls = np.zeros(kmax)
		signs = np.zeros(kmax)
		var_R = np.copy(self.R)
		var_Z = np.copy(self.Z)
		
		irrepresentable = True
		order_irrep = 0
		for k in range(kmax):
			var_R, var_Z, T, lambdas[k], indexes[k], correls[k], signs[k] = self.rec(var_R, var_Z, T)
			irrepresentable = irrepresentable and bool(np.amax(np.abs(T)) < 1)
			if irrepresentable:
				order_irrep += 1
		self.order_irrep = order_irrep	
		self.larspath = {'lambdas': lambdas, 'indexes': indexes, 'correls': correls, 'signs': signs}


	def display_lars_path(self, number_knots=10, vertical_lines_knots=False):
		lambdas = self.larspath['lambdas']
		indexes = self.larspath['indexes'].astype(int)
		signs = self.larspath['signs']
		number_knots = min(number_knots, len(lambdas))
		knot2coeffs = np.zeros((number_knots,number_knots))
		lambdas = lambdas[:number_knots]
		indexes = indexes[:number_knots]
		for k in range(1,number_knots):
			temp = self.R[indexes[:k], :]
			Minv = np.linalg.pinv(temp[:,indexes[:k]])
			knot2coeffs[k,:k] = knot2coeffs[k-1,:k] + (lambdas[k-1] - lambdas[k]) * Minv @ np.array(signs[:k])
		x = np.sum(np.abs(knot2coeffs), axis=1)
		for k in range(number_knots):
			if k<=10:
				plt.plot(x[k:], knot2coeffs[k:,k], label='{0}'.format(str(indexes[k])))
			else:
				plt.plot(x[k:], knot2coeffs[k:,k])

		plt.tight_layout()
		plt.legend(title='Predictors', bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.axhline(y=0,c='black')
		plt.xlabel(r'$\|\hat{\beta}\|_1$')
		plt.ylabel(r'Coefficients $\hat{\beta}_k$')
		if vertical_lines_knots:
			for k in range(number_knots):
				plt.axvline(x=x[k], c='black', linestyle='--')
				if k%2==0:
					plt.text(x[k], plt.ylim()[1]+0.5, str(k))
		else:
			plt.title('LAR path with the first {0} knots'.format(str(number_knots)))
		plt.show()
		return knot2coeffs




	def rec_update(self, R, Z, T, indexes, signs):
		for i,val_index in enumerate(indexes):
			# if Z[val_index] < Z[val_index + self.p]:
			# 	signed_index = int(val_index + self.p)
			# 	signs.append(-1)
			# else:
			# 	signed_index = int(val_index)
			# 	signs.append(+1)
			signed_index = int(val_index + self.p * (1-signs[i])/2)
			x_return = R[signed_index, ] / R[signed_index, signed_index]
			R = R - np.dot(x_return.reshape(np.size(x_return), 1),
							  np.asarray(R[signed_index]).reshape(1, np.size(x_return)))
			# % set to 0 the variance-covariance of the chosen index
			R[val_index, ] = np.zeros(2 * self.p)
			R[:, val_index] = np.zeros(2 * self.p)
			R[val_index + self.p, ] = np.zeros(2 * self.p)
			R[:, val_index + self.p] = np.zeros(2 * self.p)
			# % residuals
			Z = Z - Z[signed_index] * x_return
			Z[val_index] = 0
			Z[val_index + self.p] = 0
			# % parameter theta: regressed coefficient (<1 implies Irrepresentability)
			T = T + (1 - T[signed_index]) * x_return
			T[val_index] = 0
			T[val_index + self.p] = 0
		return R, Z, T, signs


	def lasso_from_lars(self, kmax=0):
		if kmax == 0:
			kmax = min(self.n, self.p, np.linalg.matrix_rank(self.R, hermitian=True))

		T = np.zeros(np.size(self.Z))
		lambdas = np.zeros(kmax)
		indexes = []
		active_set = {}
		signs = []
		var_R = np.copy(self.R)
		var_Z = np.copy(self.Z)
		knot2coeffs = np.zeros((kmax,kmax))
		var_R, var_Z, T, lambdas[0], index, __, sign = self.rec(var_R, var_Z, T)
		indexes.append(index)
		signs.append(sign)
		active_set[index] = 0
		nextactive_ind = 1
		for k in range(1,kmax):
			active_ind = list(active_set.values())
			active_items = list(active_set.items())
			temp = self.R[indexes, :]
			Minv = np.linalg.inv(temp[:,indexes])
			if len(active_set) < self.p:
				newvar_R, newvar_Z, newT, lambdas[k], newindex, __, newsign = self.rec(var_R, var_Z, T)
				knot2coeffs[k,active_ind] = knot2coeffs[k-1,active_ind] + (lambdas[k-1] - lambdas[k]) * Minv @ np.array(signs)
				indexes.append(newindex)
				active_set[newindex] = nextactive_ind
				nextactive_ind += 1
			index2remove = -1
			minknot = lambdas[k-1]
			for key,i in active_items:
				if np.dot(Minv[i,:], np.array(signs)) * knot2coeffs[k-1,i] < -1e-8:
					print('OK')
					cross_axis = lambdas[k-1] + knot2coeffs[k-1,i] /  np.dot(Minv[i,:], np.array(signs)) 
					if cross_axis < minknot and cross_axis >=0:
						minknot = cross_axis
						index2remove = key

			print('index2remove',index2remove)
			print(active_set)
			print(indexes)

			if index2remove != -1:
				lambdas[k] = minknot
				knot2coeffs[k,active_ind] = knot2coeffs[k-1, active_ind] + (lambdas[k-1] - lambdas[k]) * Minv @ np.array(signs)
				if len(indexes)>len(active_ind):
					del active_set[newindex]
					nextactive_ind -= 1
					indexes.pop(-1)
				indexes.remove(index2remove)
				var_R = np.copy(self.R)
				var_Z = np.copy(self.Z)
				T = np.zeros(np.size(self.Z))
				signs.pop(index2remove)
				var_R, var_Z, T, signs = self.rec_update(var_R, var_Z, T, indexes, signs)
				#signs = list(var_Z[indexes] / lambdas[k])
				active_set = {index: i for i,index in enumerate(indexes)}
				nextactive_ind = len(active_set)

				# signed_indexes = np.array(indexes) + (1-np.array(signs))/2 * self.p
				# signed_indexes = signed_indexes.astype(int)
				# temp = self.R[signed_indexes, :]
				# Minv = np.linalg.inv(temp[:,signed_indexes])
				# T = self.R[:,signed_indexes] @ Minv @ np.ones(len(signed_indexes))
				# for ind in indexes:
				# 	T[ind] = 0
				# 	T[ind + self.p] = 0
			elif len(active_ind) < self.p:
				var_R, var_Z, T = newvar_R, newvar_Z, newT
				signs = list(signs)
				signs.append(newsign)
			else:
				print('STOPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP')
				knot2coeffs = knot2coeffs[:k,:]
				break

			print('CONTI')
			print(active_set)
			print(indexes)
		x = np.sum(np.abs(knot2coeffs), axis=1)
		kmax = knot2coeffs.shape[0]
		for k in range(kmax):
			if k<=-1:
				plt.plot(x[k:], knot2coeffs[k:,k], label='{0}'.format(str(indexes[k])))
			else:
				plt.plot(x[k:], knot2coeffs[k:,k])

		plt.tight_layout()
		#plt.legend(title='Predictors', bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.axhline(y=0,c='black')
		plt.xlabel(r'$\|\hat{\beta}\|_1$')
		plt.ylabel(r'Coefficients $\hat{\beta}_k$')
		plt.title('LASSO path from LARS path with the first {0} knots'.format(str(kmax)))
		plt.show()
		return knot2coeffs

	# def lasso_from_lars(self, number_knots=10, vertical_lines_knots=False):
	# 	lambdas = self.larspath['lambdas']
	# 	indexes = self.larspath['indexes'].astype(int)
	# 	number_knots = min(number_knots, len(lambdas))
	# 	knot2coeffs = np.zeros((number_knots+1,number_knots))
	# 	lambdas = lambdas[:number_knots]
	# 	indexes = indexes[:number_knots]
	# 	active_set = [indexes[0]]
	# 	for k in range(1,number_knots+1):
	# 		temp = self.R[active_set, :]
	# 		Minv = np.linalg.inv(temp[:,active_set])
	# 		coeffs_k = Minv @ self.Z[active_set]
	# 		savedcodir = np.float('inf')
	# 		savedind = -1
	# 		for i, coeff in enumerate(coeffs_k):
	# 			if (coeff*knot2coeffs[k-1,i]<0):
	# 				codir = knot2coeffs[k-1,i]/(coeff-knot2coeffs[k-1,i])
	# 				if codir<savedcodir:
	# 					savedcodir = codir
	# 					savedind = i
	# 		if savedind != -1:







	# 	x = np.sum(np.abs(knot2coeffs), axis=1)
	# 	for k in range(number_knots):
	# 		if k<=10:
	# 			plt.plot(x[k:], knot2coeffs[k:,k], label='{0}'.format(str(indexes[k])))
	# 		else:
	# 			plt.plot(x[k:], knot2coeffs[k:,k])

	# 	plt.tight_layout()
	# 	plt.legend(title='Predictors', bbox_to_anchor=(1.05, 1), loc='upper left')
	# 	plt.axhline(y=0,c='black')
	# 	plt.xlabel(r'$\|\hat{\beta}\|_1$')
	# 	plt.ylabel(r'Coefficients $\hat{\beta}_k$')
	# 	if vertical_lines_knots:
	# 		for k in range(number_knots):
	# 			plt.axvline(x=x[k], c='black', linestyle='--')
	# 			if k%2==0:
	# 				plt.text(x[k], plt.ylim()[1]+0.5, str(k))
	# 	else:
	# 		plt.title('LAR path with the first {0} knots'.format(str(number_knots)))
	# 	plt.show()


	# def lars_standard(self, kmax=0):
	# 	R = np.copy(self.R)
	# 	Z = np.copy(self.Z)
	# 	lambdas = np.zeros(kmax)
	# 	indexes = np.zeros(kmax)
	# 	correls = np.zeros(kmax)
	# 	for k in range(kmax):


	def lars_projection(self, kmax=0):
		if kmax == 0:
			kmax = min(self.n, self.p, np.linalg.matrix_rank(self.R, hermitian=True))

		R = np.copy(self.R)
		Z = np.copy(self.Z)
		signed_index = np.argmax(Z)
		signed_indexes = [signed_index]
		signs = [2*(signed_index<self.p)-1]
		lambdas = [Z[signed_index]]
		T = R[:,signed_index] / R[signed_index,signed_index]
		correls = [safe_div(np.sqrt(R[signed_index, signed_index]),
						   1 - T[signed_index])]

		for k in range(1,kmax):
			if len(signed_indexes)==1:
				projector = R[:,signed_indexes] / R[signed_indexes,signed_indexes]
			else:
				temp = R[signed_indexes,:]
				projector = np.dot(R[:,signed_indexes], np.linalg.pinv(temp[:,signed_indexes]))
			projZ = np.dot(projector,  Z[signed_indexes])
			T = np.dot(projector, np.ones(k))
			ind_irrep = np.where(T<1)[0]
			knot = []
			for i in ind_irrep:
				knot.append(safe_div(Z[i]-projZ[i], 1-T[i]))
			signed_index = np.argmax(knot)
			lambdas.append(knot[signed_index])
			signed_index = ind_irrep[signed_index]
			signed_indexes.append(signed_index)
			correls.append(correls.append(safe_div(np.sqrt(R[signed_index, signed_index]),
						   1 - T[signed_index])))
			signs.append(2*(signed_index<self.p)-1)

		indexes = np.array(list(map(lambda x:int(x % self.p),signed_indexes)))
		self.larspath = {'lambdas': lambdas, 'indexes': indexes, 'correls': correls, 'signs': signs}
