#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 21:06:46 2019

@author: decastro
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from multiple_spacing_tests import lar_rec, get_order, get_residual, observed_significance_spacing, stacked_bar

class MultipleTest:
    def __init__(self):
        self.reject =[]
        self.indexes = []
        self.names = []
#%% Data 1:  DRUG = APV
X = np.loadtxt('design.txt')
y = np.loadtxt('observation.txt')
n,p = np.shape(X)
lambdas, indexes, correls, R, Z = lar_rec(X, y, normalization = True)
lars = [lambdas, indexes, correls]
order_irrep = get_order(Z,R)

#Variance estimation
t=101
R1, Z1 = get_residual(R, Z, t)
w, v = np.linalg.eig(R1)
w = np.real(w)
w2 = np.zeros(np.shape(w))
for k in range(np.size(w)):
    if abs(w[k])>1e-8:
        w2[k] = abs(w[k])**(-0.5)  
        R2 = np.real(np.dot(v,np.dot(np.diag(w2),np.transpose(v))))
        Z2 = np.dot(R2,Z1)
# We conpute the variance estimation
d = np.linalg.matrix_rank(R2, hermitian=True)
V2 = v[:,0:d]
Y2 = np.real(np.dot(np.transpose(V2),Z2))
var = (np.sum(Y2**2)-d*((np.sum(Y2)/d)**2))/(d-1)
sigma = np.sqrt(var)

# Multiple Testing
alpha = 0.2
number_hyp = order_irrep
hat_alpha = np.zeros(number_hyp)
for start in range(number_hyp):
    hat_alpha[start] = observed_significance_spacing(lars, sigma, start)
p_ordered = sorted(hat_alpha, reverse=False)
x = np.cumsum(np.ones(number_hyp))
plt.plot(x, p_ordered)
plt.plot(x, alpha*x/number_hyp)
plt.tight_layout()
plt.xlabel(r'$k$')
plt.xlim([1,24])
plt.ylim([0,0.2])
plt.ylabel(r'$p_{(k)}$')
plt.title('BH for Multiple Spacings on resistance to APV')
plt.show()

# We found that 20 hyptohese are significant
k = 20

multiple_test = MultipleTest()
hat_alpha[np.isnan(hat_alpha)] = 0 
rejected = hat_alpha<=alpha*k/number_hyp
multiple_test.reject = rejected
rej = 1*rejected
ind = indexes[:number_hyp]*rej
ind = ind[ind!=0]
ind = ind.astype(int)
multiple_test.indexes = ind
true = np.genfromtxt('tsm_true.txt', delimiter="\n", dtype=None)
names = np.genfromtxt('predictors_names.txt', dtype=str)
rejected_names = names[ind]
multiple_test.names = rejected_names

# Results:
# 13 discoveries: 13 true and 0 false discoveries (fdp = 0)
# K = 26
# n = 767 and p = 201
#%% Data 2: DRUG = ATV
X = np.loadtxt('X_ATV.txt')
y = np.loadtxt('Y_ATV.txt')
n,p = np.shape(X)
lambdas, indexes, correls, R, Z = lar_rec(X, y, normalization = True)
lars = [lambdas, indexes, correls]
order_irrep = get_order(Z,R)

#Variance estimation
t=101
R1, Z1 = get_residual(R, Z, t)
w, v = np.linalg.eig(R1)
w = np.real(w)
w2 = np.zeros(np.shape(w))
for k in range(np.size(w)):
    if abs(w[k])>1e-8:
        w2[k] = abs(w[k])**(-0.5)  
        R2 = np.real(np.dot(v,np.dot(np.diag(w2),np.transpose(v))))
        Z2 = np.dot(R2,Z1)
# We conpute the variance estimation
d = np.linalg.matrix_rank(R2, hermitian=True)
V2 = v[:,0:d]
Y2 = np.real(np.dot(np.transpose(V2),Z2))
var = (np.sum(Y2**2)-d*((np.sum(Y2)/d)**2))/(d-1)
sigma = np.sqrt(var)

# Multiple Testing
alpha = 0.2
number_hyp = order_irrep
hat_alpha = np.zeros(number_hyp)
for start in range(number_hyp):
    hat_alpha[start] = observed_significance_spacing(lars, sigma, start)
p_ordered = sorted(hat_alpha, reverse=False)
x = np.cumsum(np.ones(number_hyp))
plt.plot(x, p_ordered)
plt.plot(x, alpha*x/number_hyp)
plt.tight_layout()
plt.xlabel(r'$k$')
plt.xlim([1,13])
plt.ylim([0,0.2])
plt.ylabel(r'$p_{(k)}$')
plt.title('BH for Multiple Spacings on resistance to ATV')
plt.show()

k = 13    
multiple_test = MultipleTest()

hat_alpha[np.isnan(hat_alpha)] = 0 
rejected = hat_alpha<=alpha*k/number_hyp
multiple_test.reject = rejected
rej = 1*rejected
ind = indexes[:number_hyp]*rej
ind = ind[ind!=0]
ind = ind.astype(int)
multiple_test.indexes = ind
true = np.genfromtxt('tsm_true.txt', delimiter="\n", dtype=None)
names = np.genfromtxt('predictors_names_atv.txt', dtype=str)
rejected_names = names[ind]
multiple_test.names = rejected_names

# 12 discoveries: 10 true and 2 false discoveries
# K = 13
# n = 328 and p = 147
#%% Data 3: DRUG = IDV
X = np.loadtxt('X_IDV.txt')
y = np.loadtxt('Y_IDV.txt')
n,p = np.shape(X)
lambdas, indexes, correls, R, Z = lar_rec(X, y, normalization = True)
lars = [lambdas, indexes, correls]
order_irrep = get_order(Z,R)

#Variance estimation
t=101
R1, Z1 = get_residual(R, Z, t)
w, v = np.linalg.eig(R1)
w = np.real(w)
w2 = np.zeros(np.shape(w))
for k in range(np.size(w)):
    if abs(w[k])>1e-8:
        w2[k] = abs(w[k])**(-0.5)  
        R2 = np.real(np.dot(v,np.dot(np.diag(w2),np.transpose(v))))
        Z2 = np.dot(R2,Z1)
# We conpute the variance estimation
d = np.linalg.matrix_rank(R2, hermitian=True)
V2 = v[:,0:d]
Y2 = np.real(np.dot(np.transpose(V2),Z2))
var = (np.sum(Y2**2)-d*((np.sum(Y2)/d)**2))/(d-1)
sigma = np.sqrt(var)

# Multiple Testing
alpha = 0.2
number_hyp = order_irrep
hat_alpha = np.zeros(number_hyp)
for start in range(number_hyp):
    hat_alpha[start] = observed_significance_spacing(lars, sigma, start)
p_ordered = sorted(hat_alpha, reverse=False)
x = np.cumsum(np.ones(number_hyp))
plt.plot(x, p_ordered)
plt.plot(x, alpha*x/number_hyp)
plt.tight_layout()
plt.xlabel(r'$k$')
plt.xlim([1,22])
plt.ylim([0,0.15])
plt.ylabel(r'$p_{(k)}$')
plt.title('BH for Multiple Spacings on resistance to IDV')
plt.show()

k = 21

multiple_test = MultipleTest()

hat_alpha[np.isnan(hat_alpha)] = 0 
rejected = hat_alpha<=alpha*k/number_hyp
multiple_test.reject = rejected
rej = 1*rejected
ind = indexes[:number_hyp]*rej
ind = ind[ind!=0]
ind = ind.astype(int)
multiple_test.indexes = ind
true = np.genfromtxt('tsm_true.txt', delimiter="\n", dtype=None)
names = np.genfromtxt('predictors_names_idv.txt', dtype=str)
rejected_names = names[ind]
multiple_test.names = rejected_names

# 17 discoveries: 14 true and 3 false discoveries (fdp = 0.3)
# K = 40
# n = 825 and p = 206

#%% Data 4: DRUG = LPV
X = np.loadtxt('X_LPV.txt')
y = np.loadtxt('Y_LPV.txt')
n,p = np.shape(X)
lambdas, indexes, correls, R, Z = lar_rec(X, y, normalization = True)
lars = [lambdas, indexes, correls]
order_irrep = get_order(Z,R)

#Variance estimation
t=101
R1, Z1 = get_residual(R, Z, t)
w, v = np.linalg.eig(R1)
w = np.real(w)
w2 = np.zeros(np.shape(w))
for k in range(np.size(w)):
    if abs(w[k])>1e-8:
        w2[k] = abs(w[k])**(-0.5)  
        R2 = np.real(np.dot(v,np.dot(np.diag(w2),np.transpose(v))))
        Z2 = np.dot(R2,Z1)
# We conpute the variance estimation
d = np.linalg.matrix_rank(R2, hermitian=True)
V2 = v[:,0:d]
Y2 = np.real(np.dot(np.transpose(V2),Z2))
var = (np.sum(Y2**2)-d*((np.sum(Y2)/d)**2))/(d-1)
sigma = np.sqrt(var)

# Multiple Testing
alpha = 0.2
number_hyp = order_irrep
hat_alpha = np.zeros(number_hyp)
for start in range(number_hyp):
    hat_alpha[start] = observed_significance_spacing(lars, sigma, start)
p_ordered = sorted(hat_alpha, reverse=False)
x = np.cumsum(np.ones(number_hyp))
plt.plot(x, p_ordered)
plt.plot(x, alpha*x/number_hyp)
plt.tight_layout()
plt.xlabel(r'$k$')
plt.xlim([1,8])
plt.ylim([0,0.20])
plt.ylabel(r'$p_{(k)}$')
plt.title('BH for Multiple Spacings on resistance to LPV')
plt.show()

k = 8

multiple_test = MultipleTest()

hat_alpha[np.isnan(hat_alpha)] = 0 
rejected = hat_alpha<=alpha*k/number_hyp
multiple_test.reject = rejected
rej = 1*rejected
ind = indexes[:number_hyp]*rej
ind = ind[ind!=0]
ind = ind.astype(int)
multiple_test.indexes = ind
true = np.genfromtxt('tsm_true.txt', delimiter="\n", dtype=None)
names = np.genfromtxt('predictors_names_lpv.txt', dtype=str)
rejected_names = names[ind]
multiple_test.names = rejected_names

# 8 discoveries: 7 true and 1 false discoveries
# K = 8
# n = 515 and p = 183

#%% Data 5: DRUG = NFV
X = np.loadtxt('X_NFV.txt')
y = np.loadtxt('Y_NFV.txt')
n,p = np.shape(X)
lambdas, indexes, correls, R, Z = lar_rec(X, y, normalization = True)
lars = [lambdas, indexes, correls]
order_irrep = get_order(Z,R)

#Variance estimation
t=101
R1, Z1 = get_residual(R, Z, t)
w, v = np.linalg.eig(R1)
w = np.real(w)
w2 = np.zeros(np.shape(w))
for k in range(np.size(w)):
    if abs(w[k])>1e-8:
        w2[k] = abs(w[k])**(-0.5)  
        R2 = np.real(np.dot(v,np.dot(np.diag(w2),np.transpose(v))))
        Z2 = np.dot(R2,Z1)
# We conpute the variance estimation
d = np.linalg.matrix_rank(R2, hermitian=True)
V2 = v[:,0:d]
Y2 = np.real(np.dot(np.transpose(V2),Z2))
var = (np.sum(Y2**2)-d*((np.sum(Y2)/d)**2))/(d-1)
sigma = np.sqrt(var)

# Multiple Testing
alpha = 0.2
number_hyp = order_irrep
hat_alpha = np.zeros(number_hyp)
for start in range(number_hyp):
    hat_alpha[start] = observed_significance_spacing(lars, sigma, start)
p_ordered = sorted(hat_alpha, reverse=False)
x = np.cumsum(np.ones(number_hyp))
plt.plot(x, p_ordered)
plt.plot(x, alpha*x/number_hyp)
plt.tight_layout()
plt.xlabel(r'$k$')
plt.xlim([1,27])
plt.ylim([0,0.20])
plt.ylabel(r'$p_{(k)}$')
plt.title('BH for Multiple Spacings on resistance to NFV')
plt.show()

k = 26
    
multiple_test = MultipleTest()

hat_alpha[np.isnan(hat_alpha)] = 0 
rejected = hat_alpha<=alpha*k/number_hyp
multiple_test.reject = rejected
rej = 1*rejected
ind = indexes[:number_hyp]*rej
ind = ind[ind!=0]
ind = ind.astype(int)
multiple_test.indexes = ind
true = np.genfromtxt('tsm_true.txt', delimiter="\n", dtype=None)
names = np.genfromtxt('predictors_names_nfv.txt', dtype=str)
rejected_names = names[ind]
multiple_test.names = rejected_names

# 16 discoveries: 14 true and 2 false discoveries
# K = 56
# n = 842 and p = 207

#%% Data 6: DRUG = RTV <- False Negatives
X = np.loadtxt('X_RTV.txt')
y = np.loadtxt('Y_RTV.txt')
n,p = np.shape(X)
lambdas, indexes, correls, R, Z = lar_rec(X, y, normalization = True)
lars = [lambdas, indexes, correls]
order_irrep = get_order(Z,R)

#Variance estimation
t=101
R1, Z1 = get_residual(R, Z, t)
w, v = np.linalg.eig(R1)
w = np.real(w)
w2 = np.zeros(np.shape(w))
for k in range(np.size(w)):
    if abs(w[k])>1e-8:
        w2[k] = abs(w[k])**(-0.5)  
        R2 = np.real(np.dot(v,np.dot(np.diag(w2),np.transpose(v))))
        Z2 = np.dot(R2,Z1)
# We conpute the variance estimation
d = np.linalg.matrix_rank(R2, hermitian=True)
V2 = v[:,0:d]
Y2 = np.real(np.dot(np.transpose(V2),Z2))
var = (np.sum(Y2**2)-d*((np.sum(Y2)/d)**2))/(d-1)
sigma = np.sqrt(var)

# Multiple Testing
alpha = 0.2
number_hyp = order_irrep
hat_alpha = np.zeros(number_hyp)
for start in range(number_hyp):
    hat_alpha[start] = observed_significance_spacing(lars, sigma, start)
p_ordered = sorted(hat_alpha, reverse=False)
x = np.cumsum(np.ones(number_hyp))
plt.plot(x, p_ordered)
plt.plot(x, alpha*x/number_hyp)
plt.tight_layout()
plt.xlabel(r'$k$')
plt.xlim([1,29])
plt.ylim([0,0.20])
plt.ylabel(r'$p_{(k)}$')
plt.title('BH for Multiple Spacings on resistance to RTV')
plt.show()

k = 27
    
multiple_test = MultipleTest()

hat_alpha[np.isnan(hat_alpha)] = 0 
rejected = hat_alpha<=alpha*k/number_hyp
multiple_test.reject = rejected
rej = 1*rejected
ind = indexes[:number_hyp]*rej
ind = ind[ind!=0]
ind = ind.astype(int)
multiple_test.indexes = ind
true = np.genfromtxt('tsm_true.txt', delimiter="\n", dtype=None)
names = np.genfromtxt('predictors_names_nfv.txt', dtype=str)
rejected_names = names[ind]
multiple_test.names = rejected_names

# 21 discoveries: 16 true and 5 false discoveries 
# K = 38
# n = 793 and p = 205

#%% Data 7: DRUG = SQV
X = np.loadtxt('X_SQV.txt')
y = np.loadtxt('Y_SQV.txt')
n,p = np.shape(X)
lambdas, indexes, correls, R, Z = lar_rec(X, y, normalization = True)
lars = [lambdas, indexes, correls]
order_irrep = get_order(Z,R)

#Variance estimation
t=101
R1, Z1 = get_residual(R, Z, t)
w, v = np.linalg.eig(R1)
w = np.real(w)
w2 = np.zeros(np.shape(w))
for k in range(np.size(w)):
    if abs(w[k])>1e-8:
        w2[k] = abs(w[k])**(-0.5)  
        R2 = np.real(np.dot(v,np.dot(np.diag(w2),np.transpose(v))))
        Z2 = np.dot(R2,Z1)
# We conpute the variance estimation
d = np.linalg.matrix_rank(R2, hermitian=True)
V2 = v[:,0:d]
Y2 = np.real(np.dot(np.transpose(V2),Z2))
var = (np.sum(Y2**2)-d*((np.sum(Y2)/d)**2))/(d-1)
sigma = np.sqrt(var)

# Multiple Testing
alpha = 0.2
number_hyp = order_irrep
hat_alpha = np.zeros(number_hyp)
for start in range(number_hyp):
    hat_alpha[start] = observed_significance_spacing(lars, sigma, start)
p_ordered = sorted(hat_alpha, reverse=False)
x = np.cumsum(np.ones(number_hyp))
plt.plot(x, p_ordered)
plt.plot(x, alpha*x/number_hyp)
plt.tight_layout()
plt.xlabel(r'$k$')
plt.xlim([1,9])
plt.ylim([0,0.20])
plt.ylabel(r'$p_{(k)}$')
plt.title('BH for Multiple Spacings on resistance to SQV')
plt.show()

k = 18
    
multiple_test = MultipleTest()

hat_alpha[np.isnan(hat_alpha)] = 0 
rejected = hat_alpha<=alpha*k/number_hyp
multiple_test.reject = rejected
rej = 1*rejected
ind = indexes[:number_hyp]*rej
ind = ind[ind!=0]
ind = ind.astype(int)
multiple_test.indexes = ind
true = np.genfromtxt('tsm_true.txt', delimiter="\n", dtype=None)
names = np.genfromtxt('predictors_names_nfv.txt', dtype=str)
rejected_names = names[ind]
multiple_test.names = rejected_names

# 8 discoveries: 7 true and 1 false discoveries
# K = 10
# n = 793 and p = 205

#%%
# Results from Knockoff and BHq
comparisons = np.genfromtxt('comparisons.txt', delimiter=",", dtype=str)


#APV
plt.figure(figsize=(6, 4))
category_labels = ['Spacing BH', 'Knockoff', 'BHq']
series_labels = ['In TSM list', 'NOT in TSM list']

data = [
    [13, 19, 17],
    [0, 3, 8]
]

stacked_bar(
    data, 
    series_labels, 
    category_labels=category_labels, 
    show_values=True, 
    value_format="{:.1f}",
    y_label="HIV-1 protease positions selected",
    grid=False
)

plt.ylim([0,35])
plt.title('Resitance to APV')
plt.savefig('barAPV.png')
plt.show()

# ATV
data = [
    [10, 22, 20],
    [2, 9, 7]
]

stacked_bar(
    data, 
    series_labels, 
    category_labels=category_labels, 
    show_values=True, 
    value_format="{:.1f}",
    y_label="HIV-1 protease positions selected",
    grid=False
)

plt.ylim([0,35])
plt.title('Resitance to ATV')
plt.savefig('barATV.png')
plt.show()


# IDV
data = [
    [14, 19, 20],
    [3, 12, 10]
]

stacked_bar(
    data, 
    series_labels, 
    category_labels=category_labels, 
    show_values=True, 
    value_format="{:.1f}",
    y_label="HIV-1 protease positions selected",
    grid=False
)

plt.ylim([0,35])
plt.title('Resitance to IDV')
plt.savefig('barIDV.png')
plt.show()

# LPV
data = [
    [7, 16, 18],
    [1, 1, 3]
]

stacked_bar(
    data, 
    series_labels, 
    category_labels=category_labels, 
    show_values=True, 
    value_format="{:.1f}",
    y_label="HIV-1 protease positions selected",
    grid=False
)

plt.ylim([0,35])
plt.title('Resitance to LPV')
plt.savefig('barLPV.png')
plt.show()

# NFV
data = [
    [14, 21, 22],
    [2, 2, 7]
]

stacked_bar(
    data, 
    series_labels, 
    category_labels=category_labels, 
    show_values=True, 
    value_format="{:.1f}",
    y_label="HIV-1 protease positions selected",
    grid=False
)

plt.ylim([0,35])
plt.title('Resitance to NFV')
plt.savefig('barNFV.png')
plt.show()

# RTV
data = [
    [16, 19, 18],
    [5, 8, 8]
]


stacked_bar(
    data, 
    series_labels, 
    category_labels=category_labels, 
    show_values=True, 
    value_format="{:.1f}",
    y_label="HIV-1 protease positions selected",
    grid=False
)

plt.ylim([0,35])
plt.title('Resitance to RTV')
plt.savefig('barRTV.png')
plt.show()

# SQV
data = [
    [7, 17, 19],
    [2, 4, 11]
]

stacked_bar(
    data, 
    series_labels, 
    category_labels=category_labels, 
    show_values=True, 
    value_format="{:.1f}",
    y_label="HIV-1 protease positions selected",
    grid=False
)

plt.ylim([0,35])
plt.title('Resitance to SQV')
plt.savefig('barSQV.png')
plt.show()

# 
