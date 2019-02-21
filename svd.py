#!/usr/bin/python
import numpy as np
from scipy import linalg

A = np.array([[1, 3, 3, 2], [2, 6, 9, 5], [-1, -3, 3, 0]])
AT = np.transpose(A)
A_AT = np.dot(A, AT)
AT_A = np.dot(AT, A)

eval1, evec1 = linalg.eig(A_AT)
eval2, evec2 = linalg.eig(AT_A)

sigma = np.dot(np.transpose(evec1),np.dot(A, evec2))
sigma = sigma.round(10)
dim = min(sigma.shape)
evals_sr = np.zeros(dim)
for i in range(dim):
	evals_sr[i] = sigma[i,i]
	if sigma[i,i]<0:
		evec1[:,i]*=-1
perm = abs(eval1).argsort()[::-1]
perm2 = abs(eval2).argsort()[::-1]

print(perm)
evec1 = evec1[:,perm]
evec2 = evec2[:,perm2]
print(abs(evals_sr))
print(perm)
sigma = np.dot(np.transpose(evec1),np.dot(A, evec2))
sigma = sigma.round(10)
A_r = np.dot(evec1,np.dot(sigma,np.transpose(evec2)))

Q1, s, Q2 = linalg.svd(A)

print(eval1)
print(evec1)
print(sigma)
print(np.transpose(evec2))
print(A_r)
print("\n")
print(Q1)
print(s)
print(Q2)
