# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 11:27:42 2019

@author: Alex
"""

import numpy as np
from scipy import linalg
import os

def zeroer(A):
    for i in range(len(A)):
        try:
            for j in range(len(A[0])):
                if np.isclose(A[i,j], 0):
                    A[i,j] = 0
        except TypeError:
            if np.isclose(A[i], 0):
                A[i] = 0
                
    return A

A = np.array(([1,3,3,2],[2,6,9,5],[-1,-3,3,0]))

ATA = np.dot(np.transpose(A), A)
AAT = np.dot(A, np.transpose(A))

eigenvals1 = zeroer(linalg.eigvals(ATA))
#eigenvals2 = zeroer(linalg.eig(AAT)[0])

sigma = np.diag(np.sqrt(eigenvals1))
ATA_eigenvectors = zeroer(linalg.eig(ATA)[1])
AAT_eigenvectors = zeroer(linalg.eig(AAT)[1])

Q1 = AAT_eigenvectors
Q2 = ATA_eigenvectors

Q2T = np.transpose(Q2)

print(Q1)
print(sigma)
print(Q2T)

original = np.dot(np.dot(Q1, sigma), Q2T)