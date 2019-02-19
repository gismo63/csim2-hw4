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
print(evec1)
print(sigma)
print(evec2)
