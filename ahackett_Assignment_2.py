# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 12:03:43 2019

@author: Alex
"""

from __future__ import division

import numpy as np
import scipy as sp
from scipy import linalg
import sympy


                    


class Assignment2:
    def __init__(self, matrix, B = np.array(([]))):
        self.matrix = matrix
        self.B = B
        self.matrix_row_dim = len(self.matrix)
        self.matrix_column_dim = len(self.matrix[0])
        self.matrix_rank = 0
        self.exists_L_in = False
        self.exists_R_in = False
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
        self.rRowEch(matrix_to_t_rref=self.matrix.copy())
        self.columnbasis = np.transpose(self.rrefTranspose[~np.all(self.rrefTranspose==0, axis=1)].copy())
        print('The Column Space Basis of the Matrix is')
        print(self.columnbasis)
        
    def LNullSpaceBasis(self):
        piv_cols = []
        for i in range(len(self.rrefTranspose)):
            for j in range(len(self.rrefTranspose[0])):
                if self.rrefTranspose[i,j] >= 1:
                    piv_cols.append(j)
                    i += 1
        free_cols = sorted(set(range(piv_cols[0], piv_cols[-1] + 1)).difference(piv_cols))
        self.lnullbasis = np.transpose(self.rrefTranspose.copy())[free_cols]
        print('The Left Null Space Basis of the Matrix is')
        print(self.lnullbasis)
        
    def rowSpaceBasis(self):
        self.rowbasis = self.matrix_rRow_Ech[~np.all(self.matrix_rRow_Ech == 0, axis =1)].copy()
        print('The Row Space Basis of the Matrix is')
        print(self.rowbasis)
                    
    def nullSpaceBasis(self):
        piv_rows = []
        for i in range(len(self.matrix_rRow_Ech)):
            for j in range(len(self.matrix_rRow_Ech[0])):
                if self.matrix_rRow_Ech[i,j] >= 1:
                    piv_rows.append(j)
                    i += 1
        free_rows = sorted(set(range(piv_rows[0], piv_rows[-1] + 1)).difference(piv_rows))
        self.nullbasis = (self.matrix_rRow_Ech.copy())[free_rows]
        if free_rows == []:
            print('Null Space is Trivial, Matrix Nullity is Zero')
            if self.matrix_column_dim - self.matrix_rank:
                print('Nullity non-Zero, Some Error has Occurred')
            elif not self.matrix_column_dim - self.matrix_rank:
                print('Matrix Nullity Confirmed Zero')
                self.nullbasis = np.transpose(np.zeros(4))
        print('The Null Space Basis of the Matrix is')
        print(self.nullbasis)
        
        
        
        
        
    def outputIni(self):
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
        print('The Matrix Rank is', self.matrix_rank)
        if not self.full_rank:
            print('Matrix is Rank-Deficient')
        elif self.full_rank:
            print('Matrix is Full Rank')
        print('A Pseudoinverse Exists in General,')
        print(r'Pseudoinverse, $A^{+}$')
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
        if not self.exists_R_in:
            self.R_inv = np.dot(np.transpose(self.matrix.copy()), linalg.inv(np.dot(self.matrix.copy(), np.transpose(self.matrix.copy()))))
            print(self.R_inv)
        else:
            print('No Right Inverse Exists!!')
            
    def findLinv(self):
        if self.exists_L_in:
            self.L_inv = np.dot(linalg.inv(np.dot(np.transpose(self.matrix.copy()),self.matrix.copy())), np.transpose(self.matrix.copy()))
        else:
            print('No Left Inverse Exists!!')
        
    def pseudoInv(self):
        if self.full_rank:
            if self.matrix_row_dim < self.matrix_column_dim:
                self.pseudo_inv = np.dot(np.transpose(self.matrix.copy()) , linalg.inv(np.dot(self.matrix.copy(), np.transpose(self.matrix.copy()))))
            elif self.matrix_row_dim > self.matrix_column_dim:
                self.pseudo_inv = np.dot(linalg.inv(np.dot(np.transpose(self.matrix.copy()),self.matrix.copy())), np.transpose(self.matrix.copy()))
        elif not self.full_rank:
            u, s, vt = linalg.svd(self.matrix)
            ut = np.transpose(u)
            v = np.transpose(vt)
            for i in range(len(s)):
                if np.isclose(s[i],0):
                    s[i] = np.inf
            s_rep = 1/s
            s_dagger = np.append(np.diag(s_rep), np.zeros((len(s_rep),self.matrix_row_dim - self.matrix_column_dim)),axis=1) 
            self.s_dagger = s_dagger
            self.ut = ut
            self.v = v
            self.pseudo_inv = np.dot(np.dot(v,s_dagger),ut)
            self.pseudo_inv[np.isclose(np.abs(self.pseudo_inv),0)] = 0
            
    def rRowEch(self, matrix_to_t_rref=np.array(([]))):
        if not matrix_to_t_rref.any():
            self.matrix_rRow_Ech = self.matrix.copy()
            mat = sympy.Matrix(self.matrix_rRow_Ech.copy())
            self.matrix_rRow_Ech = np.array(mat.rref()[0]).copy()
        else:
            T_mat = sympy.Matrix(np.transpose(matrix_to_t_rref))
            self.rrefTranspose = np.array(T_mat.rref()[0]).copy()
            
        

        
    def findRank(self):
        self.rRowEch()
        self.matrix_rank = np.count_nonzero((self.matrix_rRow_Ech != 0).sum(1))
        if self.matrix_rank == self.matrix_column_dim:
            self.exists_L_in = True
        if self.matrix_rank == self.matrix_row_dim:
            self.exists_R_in = True
        if self.matrix_rank < min([self.matrix_column_dim,self.matrix_row_dim]):
            self.full_rank = False
        elif not self.matrix_rank < min([self.matrix_column_dim,self.matrix_row_dim]):
            self.full_rank = True
            
        
        
    def testSolutionExist(self):
        if not self.B.any():
            print('No B vector Found, Cannot Proceed')
        if np.allclose(np.dot(np.dot(self.matrix.copy(), self.pseudo_inv.copy()),self.B.copy()),self.B.copy()):
            print('AA^+b = b, hence, an exact solution exists')
            self.exact_solution = True
        elif ~np.allclose(np.dot(np.dot(self.matrix.copy(), self.pseudo_inv.copy()),self.B.copy()),self.B.copy()):
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
        if self.exact_solution:
            x = np.dot(self.pseudo_inv.copy(), self.B.copy())
            self.solution = x.copy()
            print('Exact Solution Complete')
            print('Exact Solution Vector')
            print(self.solution)
        elif not self.exact_solution:
            if not self.full_rank:
                print('Matrix is Rank Deficient, Using Minimum Norm Solution')
                self.minNorm()
        

    def minNorm(self):
        self.GSQR()
        if self.QR_work:
            b = self.B.copy()
            c = np.zeros(self.matrix_row_dim)
            c = np.dot((self.Q),b)
            
            #R.x = c
            x = np.zeros(self.matrix_row_dim)
            x[-1] = c[-1]
            for i in range(self.matrix_row_dim -1, -1, -1):
                x[i] = (c[i] - np.dot(self.R[i,i+1:self.matrix_row_dim].copy(), x[i+1:self.matrix_row_dim].copy()))/self.R[i,i]
            
            
            self.solution = x.copy()
            print('Minimum Norm Solution via QRD Complete')
            print('Min Norm Solution Vector')
            print(self.solution)
        elif not self.QR_work:
            self.solution = np.dot(np.dot(np.dot(self.v, self.s_dagger),self.ut),self.B)
            print('Solution via SVD Complete')
            print('Solution Vector')
            print(self.solution)
            
            
        
    def GSQR(self):
        self.Q = np.zeros((self.matrix_row_dim, self.matrix_column_dim))
        self.R = np.zeros((self.matrix_column_dim, self.matrix_column_dim))
        for j in range(1, self.matrix_column_dim):
            v = self.matrix[:,j].copy() + 0.0
            if j > 1:
                for i in range(1,j-1):
                    self.R[i,j] = np.transpose(np.dot(self.Q[:,i].copy(), self.matrix[:,j].copy()))
                    v -= self.R[i,j].copy() * self.Q[:,i].copy()
            self.R[j,j] = np.sqrt(np.sum(v*v))
            self.Q[:,j] = v.copy() / self.R[j,j]
        if np.allclose(np.dot(self.Q, self.R), self.matrix):
            print('QR Decomposition Complete')
            self.QR_work = True
        else:
            print('No Sucessful QRD Found, Resorting to SVD')
            self.QR_work = False
        
 
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

