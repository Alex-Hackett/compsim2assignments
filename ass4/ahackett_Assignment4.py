#!/usr/bin/python3

"""
PYU44C01 Linear Algebra Assignment 4
Alexander Hackett
15323791
ahackett@tcd.ie
24/02/19
"""
#imports
import numpy as np
from scipy import linalg
import os
#For producing PCA diagrams
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

def zeroer(A):
    '''
    Simple function that replaces extremely small values (within np.isclose()
    range of zero) with zero, primarily for the purposes of more asthetic 
    print output
    '''
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
    '''
    singularValueDecom() is a singular value decomposition script. Given an
    input matrix A, singularValueDecom(A) returns Q1, sigma and Q2T, the 
    three SVD matrices that when multiplied give back the original A matrix
    '''
    #Find A^T A and AA^T
    ATA = np.dot(np.transpose(A), A)
    AAT = np.dot(A, np.transpose(A))
    
    #Determine the eigenvalues and eigenvectors of A^T A and AA^T
    eigenvals_ATA, ATA_eigenvectors = (np.linalg.eig(ATA))
    eigenvals_AAT, AAT_eigenvectors = (np.linalg.eig(AAT))
    
    #Since ATA and AAT have the same eigenvalues, sort the eigenvalues 
    #of ATA according to their absolute values
    ATA_index = abs(eigenvals_ATA).argsort()[::-1]
    eigenvals_ATA = eigenvals_ATA[ATA_index]
    #Now, reorder the eigevectors of ATA according to the order of the
    #corresponding eigenvalues
    ATA_eigenvectors = ATA_eigenvectors[:,ATA_index]
    
    #Analagously to above, sort the eigenvectors of AAT according to the abs
    #values of their corresponding eigenvalues
    AAT_index = abs(eigenvals_AAT).argsort()[::-1]
    eigenvals_AAT = eigenvals_AAT[AAT_index]
    AAT_eigenvectors = AAT_eigenvectors[:,AAT_index]
    
    #Take the non-zero eigenvalues of AAT/ATA
    rows_AAT = np.nonzero(eigenvals_AAT)
    non_zero_AAT = eigenvals_AAT[rows_AAT].copy()
    
    #Construct Q1 from the sorted eigenvectors of AAT
    Q1 = AAT_eigenvectors
    #Deal with the sign Indeterminacy of the svd so results are consistent
    Q1[:,1:] = -Q1[:,1:].copy()
    #Construct the sigma matrix from the square roots of the non-zero 
    #eigenvalues of ATA/AAT
    temp = np.diag(np.sqrt(non_zero_AAT))
    #Ensure that the sigma matrix has the right dimensions to allow the matrix
    #A to be reconstructed as A = Q1.Sigma.Q2T
    sigma = np.zeros_like(A).astype(np.float64)
    if max(sigma.shape) > max(temp.shape):
        sigma[:temp.shape[0],:temp.shape[1]] = temp
    else: sigma = temp
    sigma[np.isnan(sigma)] = 0
    
    #Construct Q2 from the sorted eigenvectors of ATA
    Q2 = ATA_eigenvectors
    Q2T = np.transpose(Q2.copy())
    #print(Q1.shape)
    #print(sigma.shape)
    #print(Q2T.shape)
    
    #Finally, test if the SVD matrices are correct by reconstructing the 
    #original matrix
    original = zeroer(np.dot(np.dot(Q1, sigma), Q2T))
    if not np.allclose(A, original):
        print('!!!!SVD Failed to Reconstruct Original Matrix!!!!')
    else:
        print('SVD Sucessfully Reconstructed the Original Matrix')
    return Q1, sigma, Q2T



def QuestionOne():
    A = np.array(([1,3,3,2],[2,6,9,5],[-1,-3,3,0]))
    Q1, sigma, Q2T = singularValueDecom(A)
    print('Input Matrix')
    print(A)
    print('Reconstructed Matrix')
    print(zeroer((np.dot(Q1,np.dot(sigma, Q2T)))))
    print('Q1 Matrix')
    print(Q1)
    print('Sigma Matrix')
    print(sigma)
    print('Q2T Matrix')
    print(Q2T)
    
    
class PCA:
    '''
    Main class for performing PCA on the iris dataset, this class reads in 
    the iris dataset, manipulates it in accordance with the instructions in
    question 3 and produced PCA diagrams for the first two and first three 
    component PCAs.
    '''
    def __init__(self,filename=False):
        '''
        Constuctor method, loads in the dataset, and then calls each method
        in turn to perform the PCA steps as outlined in question 3
        '''
        if filename:
            self.irisData = np.loadtxt(filename,usecols = (0,1,2,3), delimiter=',')
            self.irisNames = np.loadtxt(filename,usecols = (4), delimiter=',', dtype='str')
        else:
            self.irisData = np.loadtxt('iris.data',usecols = (0,1,2,3), delimiter=',')
            self.irisNames = np.loadtxt('iris.data',usecols = (4), delimiter=',', dtype='str')
        self.irisData = np.array(self.irisData)
        self.datasetRows = len(self.irisData[0])
        self.datasetCols = len(self.irisData[1])
        #Center the dataset
        self.centering()
        #Find the covarience matrix
        self.coVar()
        #Find the eigendecompositon of the C matrix and use its eigenvectors
        #to determine the principal axes of the dataset
        self.eignDecomC()
        #Determine the principal componenets of the dataset
        self.princCom()
        #confirm that an SVD can be used as a more efficient method of 
        #constructing the C matrix
        self.svdCheck()
        #Produce the PCA diagrams so that the analysis can be performed.
        self.plotAnalysisDiagram()
        
    def centering(self):
        '''
        Given the iris dataset, find the mean of each column, then subtract 
        the mean of each column from all the enteries in that column, to 
        produce a centered matrix X where the mean of each column is 0
        '''
        self.X = self.irisData.copy()
        col_means = np.mean(self.irisData, axis=0)
        self.X = self.X.copy() - col_means
        #Test if the centering worked, it can be easy to confuse axis 
        #assignments/slicing!
        new_col_mean = np.mean(self.X, axis=0)
        if not np.allclose(new_col_mean,0): print('Centering Failed!!')
        
    def coVar(self):
        '''
        Directly computes the covariance matrix, C as C = X^T.X / (n-1), where
        X is the centered dataset and n is the sample size
        '''
        self.C = np.dot(np.transpose(self.X), self.X) / (self.datasetRows - 1)
        
    def eignDecomC(self):
        '''
        Perform an eigendecomposition of the covariance matrix in order to
        compute the principal directions of the dataset
        '''
        #Find the eigenvalues and eigenvectors of the covariance matrix
        eigenVals, eigenVecs = linalg.eig(self.C)
        #Construct the eigenvalue matrix Lambda, with the eigenvalues of C
        #on the diagonals
        self.Lambda = np.diag(eigenVals)
        self.V = eigenVecs
        #Compute V inverse
        self.V_inv = linalg.inv(self.V.copy())
        #Confirm that the correct matrices were computed by attempting to
        #reconstruct the covariance matrix from V, Lambda and V_inv
        reconstruct = np.dot(np.dot(self.V, self.Lambda),self.V_inv)
        if np.allclose(self.C, reconstruct):
            print('Covariance Matrix Reconstructed via EigenDecom')
        else: print('EigenDecom Failed!!')
        
    def princCom(self):
        '''
        Compute the principal componenets of the dataset by projecting the
        centered dataset X onto the principal axes, corresponding to the 
        eigenvectors V.
        '''
        self.PC = np.dot(self.X, self.V)
        
    def svdCheck(self):
        '''
        Construct the covariance matrix C using the SVD of X instead of 
        directly, this allows the covariance matrix to be constructed without
        the computationally expensive step of computing X^T.X
        '''
        #perform the svd
        U, S, VT = linalg.svd(self.X)
        S = np.diag(S)
        '''
        Construct the covariance matrix C as C = V.S^2.V^T / (n-1) where n is
        the sample size
        '''
        C_construct_svd = np.dot(np.dot(np.transpose(VT), S**2), VT) / (self.datasetRows - 1)
        #Confirm that both the svd method and the direct computation produce
        #the same covariance matrix
        if np.allclose(self.C, C_construct_svd):
            print('SVD Sucessfully Constructed the C Matrix')
        else:
            print('SVD Failed to Construct the C Matrix')
            
    def plotAnalysisDiagram(self):
        '''
        This method produces two PCA diagrams, a 2D diagram plotting the
        first two principal componenets and a 3D digram plotting the first
        three principal componenets.
        '''
        
        firstPC = self.PC[:,0]
        secondPC = self.PC[:,1]
        thirdPC = self.PC[:,2]
        
        fig1 = plt.figure()
        '''
        First figure, 2D plot of first two principal componenets
        '''
        #Plot the Iris Setosa points
        plt.plot(firstPC[0:49], secondPC[0:49], 'ro', label = 'Iris Setosa')
        #plot the iris versicolor points
        plt.plot(firstPC[50:99], secondPC[50:99], 'go', label = 'Iris Versicolor')
        #plot the iris virginica points
        plt.plot(firstPC[100:149], secondPC[100:149], 'bo', label = 'Iris Virginica')
        plt.title('PC Analysis Diagram')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.legend()
        plt.show()
        
        fig2 = plt.figure()
        '''
        Second figure, 3D plot of first 3 principal componenets
        '''
        #Set up 3D plot
        ax = fig2.add_subplot(111, projection='3d')
        #Plot the iris setosa point
        ax.scatter(firstPC[0:49], secondPC[0:49],thirdPC[0:49], 'ro', label = 'Iris Setosa')
        #plot the iris versicolor point
        ax.scatter(firstPC[50:99], secondPC[50:99],thirdPC[50:99], 'go', label = 'Iris Versicolor')
        #plot the iris virginica points
        ax.scatter(firstPC[100:149], secondPC[100:149],thirdPC[100:149], 'bo', label = 'Iris Virginica')
        plt.legend()
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        ax.set_zlabel('Third Principal Component')
        plt.show()
        
def QuestionThree():
    #Construct and hence run the default PCA constructor
    PCA1 = PCA()
        
def main():
    QuestionOne()
    QuestionThree()
    
if __name__ == '__main__':
    main()
    
        
        
                
            
        
    
