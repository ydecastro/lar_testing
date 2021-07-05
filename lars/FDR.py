
import numpy as np
import matplotlib.pyplot as plt
from .LARS import LARS
from .Knockoffs import Knockoffs
from .BHAlgos import BHAlgos



class Testing(LARS, Knockoffs, BHAlgos):

	def __init__(self):
		super(LARS, self).__init__()
		super(Knockoffs, self).__init__()
		super(BHAlgos, self).__init__()
		self.noise_correlation = 0

	def show_discoveries(self, X, y, algos, params_algos, nb_fdrs=10, names_algos=None):
		if names_algos is None:
			names_algos = algos
		target_fdr = np.linspace(0.01, 0.3, nb_fdrs)
		discoveries = np.zeros((len(algos), nb_fdrs)) 
		for i, algo in enumerate(algos):
			for j, alpha in enumerate(target_fdr):
				func = self.__getattribute__('support_fdr_'+algo)
				paras = params_algos[i]
				paras['alpha'] = alpha
				support = func(X, y, **paras)
				discoveries[i,j] = len(support)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		for i in range(len(algos)):
			plt.plot(target_fdr, discoveries[i,:], label=names_algos[i], marker=i)		
		ax.set_xlabel('Target FDR level alpha') 
		ax.set_ylabel('# of Rejections')
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		plt.show()

	def show_TP_vs_FP(self, X, y, true_support, algos, params_algos, corresponding_support=None, nb_fdrs=10, names_algos=None, save_figure=None):
		if names_algos is None:
			names_algos = algos
		target_fdr = np.linspace(0.01, 1, nb_fdrs)
		TP = np.zeros((len(algos), nb_fdrs)) 
		FP = np.zeros((len(algos), nb_fdrs)) 
		for i, algo in enumerate(algos):
			for j, alpha in enumerate(target_fdr):
				func = self.__getattribute__('support_fdr_'+algo)
				paras = params_algos[i]
				paras['alpha'] = alpha
				selected = func(X, y, **paras)
				if corresponding_support is not None:
					selected = np.array(corresponding_support)[selected]
				TP[i,j] = len(true_support) - len(set(true_support)-set(selected))
				FP[i,j] = len(set(selected)-set(true_support))
		fig = plt.figure()
		ax = fig.add_subplot(111)
		for i in range(len(algos)):
			plt.plot(TP[i,:], FP[i,:], label=names_algos[i], marker=i)		
		ax.set_xlabel('# correct discoveries (i.e. true positives)') 
		ax.set_ylabel('# incorrect discoveries (i.e. false positives)')
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		if save_figure is not None:
			plt.savefig(save_figure)
		plt.show()

	def show_fdrs_vs_powers(self, X, y, true_support, algos, params_algos, nb_fdrs=10, names_algos=None, save_figure=None):
		if names_algos is None:
			names_algos = algos
		target_fdr = np.linspace(0.01, 1, nb_fdrs)
		fdrs = np.zeros((len(algos), nb_fdrs)) 
		powers = np.zeros((len(algos), nb_fdrs)) 
		for i, algo in enumerate(algos):
			for j, alpha in enumerate(target_fdr):
				func = self.__getattribute__('fdr_power_'+algo)
				paras = params_algos[i]
				fdrs[i,j], powers[i,j] = func(X, y, true_support, alpha, **paras)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		for i in range(len(algos)):
			plt.plot(fdrs[i,:], powers[i,:], label=names_algos[i], marker=i)		
		ax.set_xlabel('FDR') 
		ax.set_ylabel('Power')
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		if save_figure is not None:
			plt.savefig(save_figure)
		plt.show()

	def show_fdrs_vs_target_fdrs(self, X, y, true_support, algos, params_algos, corresponding_support=None, nb_fdrs=10, names_algos=None, save_figure=None):
		if names_algos is None:
			names_algos = algos
		target_fdr = np.linspace(0.01, 0.4, nb_fdrs)
		fdrs = np.zeros((len(algos), nb_fdrs)) 
		for i, algo in enumerate(algos):
			for j, alpha in enumerate(target_fdr):
				func = self.__getattribute__('support_fdr_'+algo)
				paras = params_algos[i]
				paras['alpha'] = alpha
				selected = func(X, y, **paras)
				if corresponding_support is not None:
					selected = np.array(corresponding_support)[selected]
				fdrs[i,j] = self.FDR(selected, true_support)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		for i in range(len(algos)):
			plt.plot(target_fdr, fdrs[i,:], label=names_algos[i], marker=i)		
		ax.set_xlabel('Target FDR') 
		ax.set_ylabel('FDR')
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		if save_figure is not None:
			plt.savefig(save_figure)
		plt.show()

	def show_TP_FP_vs_target_fdrs(self, X, y, true_support, algos, params_algos, corresponding_support=None, nb_fdrs=10, names_algos=None, save_figure=None):
		if names_algos is None:
			names_algos = algos
		target_fdr = np.linspace(0.01, 0.4, nb_fdrs)
		TPs = np.zeros((len(algos), nb_fdrs)) 
		FPs = np.zeros((len(algos), nb_fdrs)) 
		for i, algo in enumerate(algos):
			for j, alpha in enumerate(target_fdr):
				func = self.__getattribute__('support_fdr_'+algo)
				paras = params_algos[i]
				paras['alpha'] = alpha
				selected = func(X, y, **paras)
				if corresponding_support is not None:
					selected = np.array(corresponding_support)[selected]
				TPs[i,j] = len(true_support) - len(set(true_support)-set(selected))
				FPs[i,j] = len(set(selected)-set(true_support))
		fig = plt.figure(figsize=(10, 3))
		ax = fig.add_subplot(121)
		for i in range(len(algos)):
			plt.plot(target_fdr, TPs[i,:], label=names_algos[i], marker=i)		
		ax.set_xlabel('Target FDR') 
		ax.set_ylabel('Discoveries (TPs)')
		ax = fig.add_subplot(122)
		for i in range(len(algos)):
			plt.plot(target_fdr, FPs[i,:], label=names_algos[i], marker=i)		
		ax.set_xlabel('Target FDR') 
		ax.set_ylabel('FPs')
		plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
		if save_figure is not None:
			plt.savefig(save_figure)
		plt.show()