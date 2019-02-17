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
            if np.isclose(A[i], 0) or (A[i] == np.nan):
                A[i] = 0
                
    return A

def singularValueDecom(A):
    ATA = np.dot(np.transpose(A), A)
    AAT = np.dot(A, np.transpose(A))
    
    eigenvals_ATA, ATA_eigenvectors = (np.linalg.eigh(ATA))
    eigenvals_AAT, AAT_eigenvectors = (np.linalg.eigh(AAT))
    
    ATA_index = eigenvals_ATA.argsort()[::-1]
    eigenvals_ATA = eigenvals_ATA[ATA_index]
    ATA_eigenvectors = ATA_eigenvectors[:,ATA_index]
    
    AAT_index = eigenvals_AAT.argsort()[::-1]
    eigenvals_AAT = eigenvals_AAT[AAT_index]
    AAT_eigenvectors = AAT_eigenvectors[:,AAT_index]
    
    rows_AAT = np.nonzero(eigenvals_AAT)
    non_zero_AAT = eigenvals_AAT[rows_AAT].copy()
    
    Q1 = AAT_eigenvectors
    #Q1[:,1:] = -Q1[:,1:].copy()
    temp = np.diag(np.sqrt(non_zero_AAT))
    sigma = np.zeros_like(A).astype(np.float64)
    sigma[:temp.shape[0],:temp.shape[1]] = temp
    sigma[np.isnan(sigma)] = 0
    Q2 = ATA_eigenvectors
    Q2T = np.transpose(Q2.copy())
    original = zeroer(np.dot(np.dot(Q1, sigma), Q2T))
    if not np.allclose(A, original):
        print('!!!!SVD Failed to Reconstruct Original Matrix!!!!')
    return Q1, sigma, Q2T




A = np.array(([1,3,3,2],[2,6,9,5],[-1,-3,3,0]))
Q1, sigma, Q2T = singularValueDecom(A)