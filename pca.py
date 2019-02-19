#!/usr/bin/python
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

values = np.loadtxt('iris.data', delimiter=',', usecols=[0,1,2,3])
labels = np.loadtxt('iris.data',dtype = np.str, delimiter=',', usecols=[4])

n = values.shape[1]

X = np.zeros(values.shape)

for i in range(n):
    mean = np.mean(values[:,i])
    X[:,i] = values[:,i] - mean

cov = np.dot(np.transpose(X),X)/(n-1)

evals, evecs = linalg.eig(cov)

pc = np.dot(values,evecs)

XT = np.transpose(X)
X_XT = np.dot(X, XT)
XT_X = np.dot(XT, X)

eval1, evec1 = linalg.eig(X_XT)
eval2, evec2 = linalg.eig(XT_X)

sigma = np.dot(np.transpose(evec1),np.dot(X, evec2))
sigma = sigma.round(10)

print (pc)

plt.plot(pc[:,0])
plt.plot(pc[:,1])
plt.plot(pc[:,2])
plt.plot(pc[:,3])
plt.show()
