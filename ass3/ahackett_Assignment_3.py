#!/usr/bin/python3


"""
PYU44C01 Linear Algebra Assignment 3
Alexander Hackett
15323791
ahackett@tcd.ie
09/02/19
"""

#Ensure python 2 compatability
from __future__ import division
import numpy as np
import scipy as sp
from scipy import linalg
import os
#For visualization of algorithm efficency
import matplotlib.pylab as plt

class MatEqnSol:
    '''
    Main class for solving the system of linear equations described be
    A.x = b, for a given b and (potentially) A
    '''
    def __init__(self, B, input_mat = False):
        '''
        Set up the coefficent matrix and result vector, set the default number
        of iterations for each of the algorithms, as well as the default 
        tolerance, given as the magnitude of the residual vector
        '''
        if np.all(input_mat):
            '''
            If the matrix is provided in the constructor call, just use that
            '''
            self.matrix = input_mat
        else:
            '''
            Otherwise use the matrix provided in the file
            A_matrix.txt
            '''
            self.matrix = np.loadtxt('A_matrix.txt')
        self.matrix_dims = len(self.matrix)
        self.B = B
        self.max_iterations = 1e6
        self.error_tolerance = 1e-9
        #make an initial estimate of the solution vector, just the zero 
        #vector in this case
        self.initial_guess = np.zeros(self.matrix_dims)
        self.steep_residual_mag = 0
        self.steep_solution = []
        self.steep_steps_taken = 0
        self.conj_residual_mag = 0
        self.conj_solution = []
        self.conj_steps_taken = 0
        #Produce an exact solution for comparison
        self.solveExact()
        #Set to True for print outs during each iterations of the
        #algorithm
        self.verbose = False
        #self.steepestDescent()
        #self.conjGrad()
        
    def printOutput(self):
        '''
        Prints out all the details given that both algorithms have been 
        utilized, in order to compare the results from the two algorithms
        and to ensure accuracy by comparing them to the exact solution
        '''
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
        '''
        Solve the system of linear equations exactly by inverting the 
        matrix A and solving x = A^-1.b
        '''
        self.exact_x = np.dot(linalg.inv(self.matrix), self.B)

    def steepestDescent(self):
        '''
        Implementation of the steepest descent algorithm in order to minimize
        the function f(x) = 1/2x^T.A.x - b.x in order to produce the best
        possible solution vector. Keeps track of the number of steps and
        the residual vector and error.
        '''
        x = self.initial_guess.copy()
        i = 0
        #Limit the maximum number of iterations in case the algorithm fails
        #to otherwise converge
        imax = self.max_iterations
        #Determine the initial residual vector as b - A.x0
        r = self.B - np.dot(self.matrix, x)
        #Determine the magnitude of this vector
        delta = np.dot(np.transpose(r), r)
        delta0 = delta
        while i < imax and delta > self.error_tolerance:
            '''
            Main loop for steepest descent, while our error is above tolerance
            determine the direction and length of the next step using
            the residual vector, take that step and then update the residual
            Continue until either error falls below tolerance, or we run out
            of iterations
            '''
            if self.verbose: print('Iteration', i)
            #Determine alpha as delta/(r^T . A . r)
            alpha = delta / (np.dot(np.transpose(r), np.dot(self.matrix, r)))
            #With r and alpha determing the direction of steepest gradient and
            #appropriate step lenght, take that step
            x = x + np.dot(alpha, r)
            #Recompute the new residual from the new solution vector
            r = self.B - np.dot(self.matrix, x)
            #And determine its new magnitude
            delta = np.dot(np.transpose(r), r)
            i += 1
            if i == imax:
                print(' !Warning! Max Iterations Exhausted')
        if not i == imax:
            if self.verbose: print('Residual Magnitude Tolerance Reached')
        #Store the magnitude of the residual
        self.steep_residual_mag = np.dot(np.transpose(r), r)
        #Store the solution determined by steepest descent
        self.steep_solution = x
        #Store the number of steps taken as a measure of efficiency
        self.steep_steps_taken = i
        
    def conjGrad(self):
        '''
        Implementation of the conjugate gradient method in order to minimize
        the function f(x) = 1/2x^T.A.x - b.x in order to produce the best
        possible solution vector. Keeps track of the number of steps and
        the residual vector and error.
        '''
        x = self.initial_guess.copy()
        i = 0
        imax = self.max_iterations
        #Determine the initial residual vector
        r = self.B - np.dot(self.matrix, x)
        #Set the initial direction vector to the residual vector
        d = r.copy()
        #Determine the initial magnitude of the residual vector
        delta = np.dot(np.transpose(r),r)
        delta0 = delta
        while i < imax and delta > self.error_tolerance:
            '''
            Main loop for conjugate gradient.
            While the error is above the tolerance set, determine the
            conjugate direction of the next step and the length of 
            that step, and store in the direction vector d, then 
            take that step and update the residual and direction 
            vectors for the next step though the loop
            '''
            if self.verbose: print('Iteration', i)
            #Determine alpha as Delta / (d^T.A.d)
            alpha = delta / (np.dot(np.transpose(d), np.dot(self.matrix, d)))
            #Take the step with the correct length in the direction of d
            #updating the solution vector with the new result
            x = x + np.dot(alpha, d)
            #Recompute the new residual vector
            r = self.B - np.dot(self.matrix, x)
            #And its new magnitude
            delta0 = delta
            delta = np.dot(np.transpose(r),r)
            #Compute Beta as Delta_i / Delta_i-1
            beta = delta / delta0
            #Compute the new direction vector from Beta, d_i-1
            #and the updated residual vector
            d = r + np.dot(beta, d)
            i += 1
            if i == imax:
                print('!Warning! Max Iterations Exhausted')
        if not i == imax:
            if self.verbose: print('Residual Magnitude Tolerance Reached')
        #Store the magnitude of the residual vector
        self.conj_residual_mag = np.dot(np.transpose(r), r)
        #Store the solution as determined
        #by the conjugate gradient algorithm
        self.conj_solution = x
        #Store the number of steps taken as a measure of efficiency
        self.conj_steps_taken = i
        
def main():
    '''
    Main function for completing the tasks required in the
    assignment. Creates a b vector = 
    [1,2,3,4,5,6,7,8,9,10,11,12,13] and produces
    steepest descent and conjugate gradient
    solutions for a  range of tolerances. 
    Then, the accuracies and efficiencies
    of the solutions in the lowest error
    tolerance case are displayed, and
    finally, a visualization of the
    efficiency of each algorithm as a
    function of error tolerance is produced
    '''
    #Create the b vector [1,2,3,4,5,6,7,8,9,10,11,12,13]
    B = np.array(([1,2,3,4,5,6,7,8,9,10,11,12,13]))
    #Create an array of tolerances from 1e-3 (high tolerance)
    #all the way down towards machine precison (1e-16)
    tolerances = np.linspace(1e-3, 1e-16, 1000)
    steep_steps = []
    conj_steps = []
    for i in tolerances:
        #Create a MatEqnSol instance with
        #the solution vector b
        s1 = MatEqnSol(B)
        #Set the error tolerance
        #(Magnitude of the residual vector)
        #To the currently selected tolerance
        s1.error_tolerance = i
        #Find a solution using the steepest
        #descent algorithm
        s1.steepestDescent()
        #Find a solution using the conjugate
        #gradient algorithm
        s1.conjGrad()
        #Keep track of the number of steps taken
        #To reach tolerance for each of the
        #two algorithms
        steep_steps.append(s1.steep_steps_taken)
        conj_steps.append(s1.conj_steps_taken)

    #Print the outcome of the lowest tolerance
    #solutions
    s1.printOutput()
        
    print('Provided b Vector')
    print(s1.B)
    
    print('b Vector Recreated Through Exact Inverse')
    print(np.dot(s1.matrix, s1.exact_x))
    
    print('b Vector Recreated Through Steepest Descent Algorithm')
    print(np.dot(s1.matrix, s1.steep_solution))
    
    print('b Vector Recreated Through Conjugate Gradient Algorithm')
    print(np.dot(s1.matrix, s1.conj_solution))
    #Create a plot comparing the efficencies
    #of each of the algorithms as a function
    #of the tolerance
    fig1 = plt.figure()
    plt.plot(tolerances, steep_steps, color = 'b', label = 'Steepest Descent')
    plt.title('Efficiency Comparison Between Steepest Descent and Conjugate Gradient')
    plt.xlabel('Acceptable Magnitude of Residual Vector')
    plt.ylabel('Number of Iterations Required to Reach Desired Residual')
    plt.plot(tolerances, conj_steps, color = 'r', label='Conjugate Gradient')
    plt.grid(which='both')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
