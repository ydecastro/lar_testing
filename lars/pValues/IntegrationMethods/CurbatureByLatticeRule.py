import numpy as np
# %% MCQMC Fast-CBC method

# % closest prime less than n
def prime(n):
	test = n
	while not isPrime(test):
		test -= 1
	return int(test)


# % prime test
def isPrime(n):
	for i in range(2, int(n ** 0.5) + 1):
		if n % i == 0:
			return False
	return True


# % vector generating the lattice for cubature
def fast_rank_1(n,
				s_max,
				omega=lambda x: x ** 2 - x + 1 / 6,
				gamma=0.9,
				print_CBC=False):
	# % n has to be prime
	if isPrime(n):
		n_prime = n
	else:
		n_prime = int(prime(n))
		if print_CBC:
			print("Warning: number of points is not a prime number, changed to n=%s points" % n_prime)
	# % generating vector
	z = np.zeros(s_max)
	#	e2  = np.zeros(s_max)

	m = int((n_prime - 1) / 2)
	q = np.ones(m)
	q0 = 1

	gamma_vec = np.cumprod(gamma * np.ones(s_max))
	beta_vec = 1 + gamma_vec / 3
	#	cumbeta	 = np.cumprod(beta_vec)

	g = np.random.randint(2, n_prime)
	perm = np.zeros(m)
	temp = 1

	for j in range(m):
		perm[j] = int(temp)
		temp = temp * g % n_prime
	perm = np.minimum(perm, n_prime - perm)
	psi = omega(perm / n_prime)
	psi0 = omega(0)
	fft_psi = np.fft.fft(psi)

	for s in range(s_max):
		E2 = np.fft.ifft(fft_psi * np.fft.fft(q))
		E2 = np.real(E2)
		w = np.argmin(E2)
		#		min_E2  = E2[w]
		if s == 0:
			w = 1
		#			noise=np.abs(E2[0]-min_E2)
		z[s] = perm[w]
		#		e2[s]   = -cumbeta[s]+(beta_vec[s]*(q0+2*np.sum(q))+gamma_vec[s]*(psi0*q0+2*min_E2))/n_prime
		ppsi = np.concatenate((psi[w::-1], psi[:w:-1]), axis=0)
		q = (beta_vec[s] + gamma_vec[s] * ppsi) * q
		q0 = (beta_vec[s] + gamma_vec[s] * psi0) * q0
		if print_CBC:
			print("s=%s, z=%s, w=%s" % (s, z[s], w))

	return [z, n_prime]