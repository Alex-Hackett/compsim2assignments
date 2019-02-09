#!/usr/bin/python3


"""
PYU44C01 Linear Algebra Assignment 3
Alexander Hackett
15323791
ahackett@tcd.ie
09/02/19
"""

from __future__ import division
import numpy as np
import scipy as sp
from scipy import linalg
import os
import matplotlib.pylab as plt

class MatEqnSol:
    def __init__(self, B, input_mat = False):
        if np.all(input_mat):
            self.matrix = input_mat
        else:
            self.matrix = np.loadtxt('A_matrix.txt')
        self.matrix_dims = len(self.matrix)
        self.B = B
        self.max_iterations = 1e6
        self.error_tolerance = 1e-9
        self.initial_guess = np.zeros(self.matrix_dims)
        self.steep_residual_mag = 0
        self.steep_solution = []
        self.steep_steps_taken = 0
        self.conj_residual_mag = 0
        self.conj_solution = []
        self.conj_steps_taken = 0
        self.solveExact()
        self.verbose = False
        #self.steepestDescent()
        #self.conjGrad()
        
    def printOutput(self):
        print('The Following A Matrix was Provided:')
        print(self.matrix)
        print('The Following B Vector was Utilized:')
        print(self.B)
        print('\n')
        print('Utilizing an Error Tolerance of:')
        print(self.error_tolerance)
        print('The Steepest Descent Algorithm Produced a Solution Vector x:')
        print(self.steep_solution)
        print('In ',self.steep_steps_taken,' Iterations')
        print('\n')
        print('The Conjugate Gradient Algorithm Produced a Solution Vector x:')
        print(self.conj_solution)
        print('In ',self.conj_steps_taken,' Iterations')
        print('\n')
        print('The Exact Solution Vector x, Produced by Inverting A was:')
        print(self.exact_x)
        if np.allclose(self.exact_x, self.steep_solution):
            print('The Steepest Descent Solution Matches the Exact Solution')
        else: print('Steepest Descent Failed to Produce the Exact Solution')
        if np.allclose(self.exact_x, self.conj_solution):
            print('The Conjugate Gradient Solution Matches the Exact Solution')
        else: print('Conjugate Gradient Failed to Produce the Exact Solution')
            
    
    def solveExact(self):
        self.exact_x = np.dot(linalg.inv(self.matrix), self.B)

    def steepestDescent(self):
        x = self.initial_guess.copy()
        i = 0
        imax = self.max_iterations
        r = self.B - np.dot(self.matrix, x)
        delta = np.dot(np.transpose(r), r)
        delta0 = delta
        while i < imax and delta > self.error_tolerance:
            if self.verbose: print('Iteration', i)
            alpha = delta / (np.dot(np.transpose(r), np.dot(self.matrix, r)))
            x = x + np.dot(alpha, r)
            r = self.B - np.dot(self.matrix, x)
            delta = np.dot(np.transpose(r), r)
            i += 1
            if i == imax:
                print(' !Warning! Max Iterations Exhausted')
        if not i == imax:
            if self.verbose: print('Residual Magnitude Tolerance Reached')
        self.steep_residual_mag = np.dot(np.transpose(r), r)
        self.steep_solution = x
        self.steep_steps_taken = i
        
    def conjGrad(self):
        x = self.initial_guess.copy()
        i = 0
        imax = self.max_iterations
        r = self.B - np.dot(self.matrix, x)
        d = r.copy()
        delta = np.dot(np.transpose(r),r)
        delta0 = delta
        while i < imax and delta > self.error_tolerance:
            if self.verbose: print('Iteration', i)
            alpha = delta / (np.dot(np.transpose(d), np.dot(self.matrix, d)))
            x = x + np.dot(alpha, d)
            r = self.B - np.dot(self.matrix, x)
            delta0 = delta
            delta = np.dot(np.transpose(r),r)
            beta = delta / delta0
            d = r + np.dot(beta, d)
            i += 1
            if i == imax:
                print('!Warning! Max Iterations Exhausted')
        if not i == imax:
            if self.verbose: print('Residual Magnitude Tolerance Reached')
        self.conj_residual_mag = np.dot(np.transpose(r), r)
        self.conj_solution = x
        self.conj_steps_taken = i
        
        
B = np.array(([1,2,3,4,5,6,7,8,9,10,11,12,13]))
tolerances = np.linspace(1e-3, 1e-15, 20000)
steep_steps = []
conj_steps = []
for i in tolerances:
    s1 = MatEqnSol(B)
    s1.error_tolerance = i
    s1.steepestDescent()
    s1.conjGrad()
    steep_steps.append(s1.steep_steps_taken)
    conj_steps.append(s1.conj_steps_taken)
fig1 = plt.figure()
plt.plot(tolerances, steep_steps, color = 'b', label = 'Steepest Descent')
plt.title('Efficiency Comparison Between Steepest Descent and Conjugate Gradient')
plt.xlabel('Acceptable Magnitude of Residual Vector')
plt.ylabel('Number of Iterations Required to Reach Desired Residual')
plt.plot(tolerances, conj_steps, color = 'r', label='Conjugate Gradient')
plt.grid(which='both')
plt.legend()
plt.show()

s1.printOutput()