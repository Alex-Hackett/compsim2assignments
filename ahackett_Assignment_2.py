#!/usr/bin/python3


"""
PYU44C01 Linear Algebra Assignment 2
Alexander Hackett
15323791
ahackett@tcd.ie
03/02/19
"""
#Ensure backward/forward compatibility
from __future__ import division
#Check for version
import six 

import numpy as np
from scipy import linalg
#To place matricies in reduced row echelon form
#Sympy is not a default install, so check if it exists,
#if not, use pip to install it locally
import os
try:
    import sympy
except ImportError:
    print('sympy is not installed locally, installing via pip')
    if six.PY2:
        os.system('python2 -m pip install sympy --user')
    elif six.PY3:
        os.system('python3 -m pip install sympy --user')
    import sympy


                    


class Assignment2:
    '''
    Class that takes an input matrix, and optionally a result vector
    and performs all the maniputations required for each problem
    '''
    def __init__(self, matrix, B = np.array(([]))):
        self.matrix = matrix
        self.B = B
        #Store the number of rows and columns, in case they differ
        self.matrix_row_dim = len(self.matrix)
        self.matrix_column_dim = len(self.matrix[0])
        #Initialize rank and existance of left/right inverse
        self.matrix_rank = 0
        self.exists_L_in = False
        self.exists_R_in = False
        #find the rank of the matrix
        self.findRank()
        
    def probOne(self):
        self.outputIni()
        self.pseudoInv()
        self.whichInv()
        self.compareScipyPinv()
        
    def probTwo(self):
        self.outputIni()
        self.pseudoInv()
        self.whichInv()
        self.testSolutionExist()
        self.solveSystem()
        print('Scipy Least Squares Solution')
        print(linalg.lstsq(self.matrix, self.B)[0])
        
        
        
    def probThree(self):
        self.outputIni()
        self.pseudoInv()
        self.whichInv()
        self.columnSpaceBasis()
        self.LNullSpaceBasis()
        self.rowSpaceBasis()
        self.nullSpaceBasis()
    
    def columnSpaceBasis(self):
        '''
        Finds A basis for the column space of the matrix
        Reduce the matrix to reduced row echelon form and identify the pivot elements
        construct a column space basis from the columns in the ORIGINAL
        matrix that correspond to the columns containing the pivot elements
        in the reduced row echelon form
        '''
        #self.rRowEch(matrix_to_t_rref=self.matrix.copy())
        self.columnbasis = self.matrix_rRow_Ech[~np.all(self.matrix_rRow_Ech==0, axis=1)].copy()
        piv_rows = []
        for i in range(len(self.matrix_rRow_Ech)):
            for j in range(len(self.matrix_rRow_Ech[0])):
                if self.matrix_rRow_Ech[i,j] >= 1:
                    piv_rows.append(j)
                    j += 1
        self.columnbasis = self.matrix[:,piv_rows]
        print('The Column Space Basis of the Matrix is')
        print(self.columnbasis)
        
    def LNullSpaceBasis(self):
        '''
        Finds A basis for the left null space of the matrix
        The left null space of the matrix is the kernal (null space) of the
        transpose of the matrix, so append the identity matrix to the right
        of the original matrix and bring to reduced row echelon form, this will
        destroy the left null space, but the identity matrix will record the 
        elementary row operations, so the row(s) of this matrix that give the
        zero matrix when used to permute the original matrix form a basis for
        the left null space, since left multiplication operates on the rows
        of a matrix in the same way that right multiplication operates on the
        columns
        '''
        A_matrix = self.matrix.copy()
        E_matrix = np.eye(max([self.matrix_row_dim, self.matrix_column_dim]))
        aug_matrix = np.hstack((A_matrix, E_matrix))
        aug_matrix = sympy.Matrix(aug_matrix)
        aug_matrix_rref = np.array(aug_matrix.rref()[0]).copy()
        E_matrix = aug_matrix_rref[:, self.matrix_column_dim:].copy()
        #print(aug_matrix)
        #print(aug_matrix_rref)
        #print(A_matrix)
        #print(E_matrix)
        R_matrix = np.dot(E_matrix, A_matrix)
        #print(R_matrix)
        zero_rows_R = [np.all(R_matrix == 0, axis =1)]
        #print(zero_rows_R)
        self.lnullbasis = E_matrix[tuple(zero_rows_R)].copy()
        print('The Left Null Space Basis of the Matrix is')
        print(self.lnullbasis)
        
    def rowSpaceBasis(self):
        '''
        Finds A row space basis for the matrix, by reducing the matrix to
        reduced row echelon form and directly forming a basis from the non-
        zero rows in this representation. This can be done since elementary
        row operations preserve the row space
        '''
        self.rowbasis = self.matrix_rRow_Ech[~np.all(self.matrix_rRow_Ech == 0, axis =1)].copy()
        print('The Row Space Basis of the Matrix is')
        print(self.rowbasis)
                    
    def nullSpaceBasis(self):
        '''
        Finds A null space basis for the matrix. A set of null space vectors
        can be constructed from the free columns of the reduced row echelon
        form of the matrix, that is, those columns that do not contain a 
        pivot element. 
        '''
        piv_rows = []
        for i in range(len(self.matrix_rRow_Ech)):
            for j in range(len(self.matrix_rRow_Ech[0])):
                if self.matrix_rRow_Ech[i,j] >= 1:
                    piv_rows.append(j)
                    j += 1
        free_rows = sorted(set(range(piv_rows[0], piv_rows[-1] + 1)).difference(piv_rows))
        #print(free_rows)
        #print(piv_rows)
        self.nullbasis = (self.matrix_rRow_Ech.copy())[free_rows]
        '''
        #If there are no free variables, then the null space is spanned 
        #by just the zero vector (trivial null space)
        if this is the case, we test if the nullity of the matrix should really
        be zero, via the rank nullity theorem, to see if a mistake
        exists somewhere in the code
        '''
        if free_rows == []:
            print('Null Space is Trivial, Matrix Nullity is Zero')
            if self.matrix_column_dim - self.matrix_rank:
                print('Nullity non-Zero, Some Error has Occurred')
            elif not self.matrix_column_dim - self.matrix_rank:
                '''
                If the nullity of the matrix should indeed be zero, then
                replace the null space basis vector with a zero vector
                of the right dimensions, rather than the null vector
                that the code produces
                '''
                print('Matrix Nullity Confirmed Zero')
                self.nullbasis = np.transpose(np.zeros(self.matrix_row_dim))
        print('The Null Space Basis of the Matrix is')
        print(self.nullbasis)
        
        
        
        
        
    def outputIni(self):
        '''
        Print information fed into class
        '''
        print('***********************************************************')
        print('***********************************************************')
        print('***********************************************************')
        print('BEGINNING COMPUTATION')
        print('***********************************************************')
        print('***********************************************************')
        print('***********************************************************')
        print('The Following Matrix, A, was Provided')
        print(self.matrix)
        if self.B.any():
            print('The Following Result Vector, B, was Provided')
            print(self.B)
            
    
    def compareScipyPinv(self):
        if not self.matrix_column_dim == self.matrix_row_dim:
            if self.full_rank:
                print('Matrix is Full Rank, non-square')
                print('scipy computes p-inverse via SVD')
                print(linalg.pinv(self.matrix.copy()))
                print('Determining the p-inverse directly gives:')
                print(self.pseudo_inv)
    
    def whichInv(self):
        '''
        Determine which, if any, inverses exist, and compute
        and print them
        '''
        print('The Matrix Rank is', self.matrix_rank)
        if not self.full_rank:
            print('Matrix is Rank-Deficient')
        elif self.full_rank:
            print('Matrix is Full Rank')
        print('A Pseudoinverse Exists in General,')
        print(r'Pseudoinverse, $A^{+}$')
        #find the generalized inverse (Moore-Penrose)
        print(self.pseudo_inv)
        if self.exists_L_in:
            print('A left Inverse Exists:')
            self.findLinv()
            print(self.L_inv)
        else:
            print('No Left Inverse Exists!')
        if self.exists_R_in:
            print('A Right Inverse Exists:')
            self.findRinv()
            print(self.R_inv)
        else:
            print('No Right Inverse Exists')
            
    def findRinv(self):
        '''
        If a right inverse exists, find it according to the formula
        A^-1R = A^T (AA^T)^-1
        '''
        if self.exists_R_in:
            self.R_inv = np.dot(np.transpose(self.matrix.copy()), linalg.inv(np.dot(self.matrix.copy(), np.transpose(self.matrix.copy()))))
            #print(self.R_inv)
        else:
            print('No Right Inverse Exists!!')
            
    def findLinv(self):
        '''
        If a left inverse exists, find it according to the formula
        A^-1L = (A^T A)^-1 A^T
        '''
        if self.exists_L_in:
            self.L_inv = np.dot(linalg.inv(np.dot(np.transpose(self.matrix.copy()),self.matrix.copy())), np.transpose(self.matrix.copy()))
        else:
            print('No Left Inverse Exists!!')
        
    def pseudoInv(self):
        '''
        Find the generalized Moore-Penrose Pseudoinverse, using an appropriate
        method, either directly if the matrix is full rank, or via a singular
        value decomposition if not
        '''
        if self.full_rank:
            '''
            If the matrix is full rank and wide (more columns than rows) then 
            the pseudoinverse is the right inverse, otherwise, if the matrix 
            is skinny, the pseudoinverse is the left inverse
            '''
            if self.matrix_row_dim < self.matrix_column_dim:
                self.pseudo_inv = np.dot(np.transpose(self.matrix.copy()) , linalg.inv(np.dot(self.matrix.copy(), np.transpose(self.matrix.copy()))))
            elif self.matrix_row_dim > self.matrix_column_dim:
                self.pseudo_inv = np.dot(linalg.inv(np.dot(np.transpose(self.matrix.copy()),self.matrix.copy())), np.transpose(self.matrix.copy()))
        elif not self.full_rank:
            '''
            If the matrix is not full rank, then a pseudoinverse can be found
            via singular value decomposition
            If A = U * Sigma * V^T, the
            A^+ = V * Sigma^Dagger * U^T
            '''
            u, s, vt = linalg.svd(self.matrix)
            ut = np.transpose(u)
            v = np.transpose(vt)
            #Replace the zero singular values with infinity to allow the
            #reciprocal of the s vector to be constructed
            for i in range(len(s)):
                if np.isclose(s[i],0):
                    s[i] = np.inf
            s_rep = 1/s
            #Construct s_dagger, the diagonal matrix with the reciprocal values
            #of s on the diagonals
            s_dagger = np.append(np.diag(s_rep), np.zeros((len(s_rep),self.matrix_row_dim - self.matrix_column_dim)),axis=1) 
            self.s_dagger = s_dagger
            self.ut = ut
            self.v = v
            #Compute the pseudoinverse
            self.pseudo_inv = np.dot(np.dot(v,s_dagger),ut)
            self.pseudo_inv[np.isclose(np.abs(self.pseudo_inv),0)] = 0
            
    def rRowEch(self):
        '''
        Produce the reduced row echelon form of a matrix, or of the transpose
        of a matrix (if matrix_to_t_rref is passed a matrix) and store in the
        appropriate var
        '''
        self.matrix_rRow_Ech = self.matrix.copy()
        mat = sympy.Matrix(self.matrix_rRow_Ech.copy())
        self.matrix_rRow_Ech = np.array(mat.rref()[0]).copy()
        

        
    def findRank(self):
        '''
        Find the rank of a matrix
        '''
        #Make a rref version of the matrix
        self.rRowEch()
        #The row rank of the matrix is the number of non-zero rows in the rref
        #representation of the matrix
        #It can be proved that the row and column ranks of a matrix are 
        #always equal
        self.matrix_rank = np.count_nonzero((self.matrix_rRow_Ech != 0).sum(1))
        if self.matrix_rank == self.matrix_column_dim:
            #If the rank equals the number of columns, then a left inverse can
            #be constructed
            self.exists_L_in = True
        if self.matrix_rank == self.matrix_row_dim:
            #If the rank equals the number of rows, then a right inverse
            #can be constructed
            self.exists_R_in = True
        if self.matrix_rank < min([self.matrix_column_dim,self.matrix_row_dim]):
            #If the rank is less than the number of rows or columns (whichever
            #is smaller), then the matrix is rank deficient
            self.full_rank = False
        elif not self.matrix_rank < min([self.matrix_column_dim,self.matrix_row_dim]):
            #Otherwise, the matrix is full rank
            self.full_rank = True
            
        
        
    def testSolutionExist(self):
        '''
        Tests if an EXACT solution exists for a system of linear equations
        A.x = b
        '''
        if not self.B.any():
            print('No B vector Found, Cannot Proceed')
        if np.allclose(np.dot(np.dot(self.matrix.copy(), self.pseudo_inv.copy()),self.B.copy()),self.B.copy()):
            #If the pseudoinverse of the matrix is exact, (i.e is a left or right inverse)
            #Then there will exist at least one exact solution
            print('AA^+b = b, hence, an exact solution exists')
            self.exact_solution = True
        elif ~np.allclose(np.dot(np.dot(self.matrix.copy(), self.pseudo_inv.copy()),self.B.copy()),self.B.copy()):
            #Otherwise, if the matrix is rank deficient, the pseudoinverse is not
            #an exact inverse, and hence, no exact solution will exist
            print('No Exact Solution Exists to this System')
            self.exact_solution = False
        if  not self.full_rank:
            print('This Matrix is Rank Deficient, Rank is')
            print(self.matrix_rank)
            print('Full Rank Would be')
            print(min([self.matrix_column_dim, self.matrix_row_dim]))
        elif self.full_rank:
            print('This Matrix is Full Rank')
            print('The Rank of This Matrix is')
            print(self.matrix_rank)
    
    def solveSystem(self):
        '''
        Solve a system of linear equations based on the
        most appropriate method
        '''
        if self.exact_solution:
            '''
            If an exact solution exists, then the pseudoinverse exists and
            is an exact inverse (left), so dot the pseudoinverse and the
            resultant matrix to exactly solve the system
            '''
            x = np.dot(self.pseudo_inv.copy(), self.B.copy())
            self.solution = x.copy()
            print('Exact Solution Complete')
            print('Exact Solution Vector')
            print(self.solution)
        elif not self.exact_solution:
            '''
            Not Implemented yet!! (WIP)
            if self.full_rank:
                self.leastSquareSolve()
            '''
            if not self.full_rank:
                '''
                If the matrix is rank deficient and has no exact solution
                a minimum norm solution is required
                '''
                print('Matrix is Rank Deficient, Using Minimum Norm Solution')
                self.minNorm()
        

    def minNorm(self):
        '''
        Solve a system of linear equations with no exact solution where
        A is rank deficient. Attempt first to perform a QR decomposition
        if that fails, use SVD instead
        '''
        #Perform the Gram-Schmit Orthogonalization QR decomposition algo
        self.GSQR()
        if self.QR_work:
            '''
            if QR decomposition worked, solve the system
            of equations by multiplying the orthogonal matrix with b,
            then solving via back subsitution
            '''
            b = self.B.copy()
            c = np.zeros(self.matrix_row_dim)
            c = np.dot((self.Q),b)
            
            #R.x = c
            #Solve via back subsitution
            x = np.zeros(self.matrix_row_dim)
            x[-1] = c[-1] #Trivial
            #Loop backwards through rows, eliminating
            for i in range(self.matrix_row_dim -1, -1, -1):
                #Divide by R[i,i] to ensure normalization
                x[i] = (c[i] - np.dot(self.R[i,i+1:self.matrix_row_dim].copy(), x[i+1:self.matrix_row_dim].copy()))/self.R[i,i]
            
            
            self.solution = x.copy()
            print('Minimum Norm Solution via QRD Complete')
            print('Min Norm Solution Vector')
            print(self.solution)
        elif not self.QR_work:
            '''
            If QR decomposition didn't work, solve directly using the SVD variables
            computed from before
            x = v.sigma_dagger.u^T.b
            will minimlize residuals on x
            '''
            self.solution = np.dot(np.dot(np.dot(self.v, self.s_dagger),self.ut),self.B)
            print('Solution via SVD Complete')
            print('Solution Vector')
            print(self.solution)
            
            
        
    def GSQR(self):
        '''
        Use GS-Orthognlization to attempt a QR decomposition of the matrix A
        '''
        #Initilize Q and R
        self.Q = np.zeros((self.matrix_row_dim, self.matrix_column_dim))
        self.R = np.zeros((self.matrix_column_dim, self.matrix_column_dim))
        #Loop through columns of matrix, placing float values in vector v
        for j in range(1, self.matrix_column_dim):
            v = self.matrix[:,j].copy() + 0.0
            #skip first column
            if j > 1:
                #Loop through remaining rows
                for i in range(1,j-1):
                    #R = Q^T.A = (Q.A)^T since Q is orthognonal
                    self.R[i,j] = np.transpose(np.dot(self.Q[:,i].copy(), self.matrix[:,j].copy()))
                    #Reduce by non-orthogonal component
                    v -= self.R[i,j].copy() * self.Q[:,i].copy()
            #Sum least squares
            self.R[j,j] = np.sqrt(np.sum(v*v))
            #Normalize Q via least squares
            self.Q[:,j] = v.copy() / self.R[j,j]
        if np.allclose(np.dot(self.Q, self.R), self.matrix):
            #test to see if the decomposition worked
            print('QR Decomposition Complete')
            self.QR_work = True
        else:
            print('No Sucessful QRD Found, Resorting to SVD')
            self.QR_work = False
        
 
def main():
    A1 = np.array(([6,1,5],[5,1,4],[0,5,-5],[2,2,0]))
    A2 = np.array(([6,1,5],[5,1,4],[1,5,-5],[2,2,0]))
    x1 = Assignment2(A1)
    x2 = Assignment2(A2)
    
    x1.probOne()
    x2.probOne()
    
    A3 = np.array(([3,6,-3,2],[2,5,0,4],[3,9,3,-1],[1,2,-1,1]))
    B3 = np.array(([3,1,-3,2]))
    x3 = Assignment2(A3, B3)
    
    x3.probTwo()
    
    A4 = np.array(([1,6,-3,0],[0,4,2,-3],[3,18,1,-5],[2,0,0,3],[2,8,2,0]))
    x4 = Assignment2(A4)
    
    x4.probThree()
    
if __name__ == '__main__':
    main()


