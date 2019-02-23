#!/usr/bin/python3

import numpy as np
from scipy import linalg
import os
import matplotlib.pylab as plt

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
    Q1[:,1:] = -Q1[:,1:].copy()
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



def QuestionOne():
    A = np.array(([1,3,3,2],[2,6,9,5],[-1,-3,3,0]))
    Q1, sigma, Q2T = singularValueDecom(A)
    print('Input Matrix')
    print(A)
    print('Reconstructed Matrix')
    print(zeroer((np.dot(Q1,np.dot(sigma, Q2T)))))
    
    
class PCA:
    def __init__(self,filename=False):
        if filename:
            self.irisData = np.loadtxt(filename,usecols = (0,1,2,3), delimiter=',')
            self.irisNames = np.loadtxt(filename,usecols = (4), delimiter=',', dtype='str')
        else:
            self.irisData = np.loadtxt('iris.data',usecols = (0,1,2,3), delimiter=',')
            self.irisNames = np.loadtxt('iris.data',usecols = (4), delimiter=',', dtype='str')
        self.irisData = np.array(self.irisData)
        self.datasetRows = len(self.irisData[0])
        self.datasetCols = len(self.irisData[1])
        self.centering()
        self.coVar()
        self.eignDecomC()
        self.princCom()
        #self.svdCheck()
        self.plotAnalysisDiagram()
        
    def centering(self):
        self.X = self.irisData.copy()
        col_means = np.mean(self.irisData, axis=0)
        self.X = self.X.copy() - col_means
        new_col_mean = np.mean(self.X, axis=0)
        if not np.allclose(new_col_mean,0): print('Centering Failed!!')
        
    def coVar(self):
        self.C = np.dot(np.transpose(self.X), self.X) / (self.datasetRows - 1)
        
    def eignDecomC(self):
        eigenVals, eigenVecs = linalg.eig(self.C)
        self.Lambda = np.diag(eigenVals)
        self.V = eigenVecs
        self.V_inv = linalg.inv(self.V.copy())
        reconstruct = np.dot(np.dot(self.V, self.Lambda),self.V_inv)
        if np.allclose(self.C, reconstruct):
            print('Covariance Matrix Reconstructed via EigenDecom')
        else: print('EigenDecom Failed!!')
        
    def princCom(self):
        self.PC = np.dot(self.X, self.V)
        
    def svdCheck(self):
        U, S, VT = singularValueDecom(self.X)
        C_construct_svd = np.dot(np.dot(np.transpose(VT), S**2), VT) / (self.datasetRows - 1)
        if np.allclose(self.C, C_construct_svd):
            print('SVD Sucessfully Constructed the C Matrix')
            
    def plotAnalysisDiagram(self):
        firstPC = self.PC[:,0]
        secondPC = self.PC[:,1]
        
        fig1 = plt.figure()
        plt.plot(firstPC[0:49], secondPC[0:49], 'ro', label = 'Iris Setosa')
        plt.plot(firstPC[50:99], secondPC[50:99], 'go', label = 'Iris Versicolor')
        plt.plot(firstPC[100:149], secondPC[100:149], 'bo', label = 'Iris Virginica')
        plt.title('PC Analysis Diagram')
        plt.xlabel('Sepal Length Direction')
        plt.ylabel('Sepal Width Direction')
        plt.legend()
        plt.show()
        
                
def QuestionTwo():
    PCA1 = PCA()
        
def main():
    QuestionOne()
    QuestionTwo()
    
if __name__ == '__main__':
    main()
    
        
        
                
            
        
    
