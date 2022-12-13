# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 01:18:12 2022

@author: Lochlann
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def preprocess_data(filename):
    df_performance = pd.read_csv(filename)
    features = df_performance.drop(columns=["Heating load", "Cooling load"], axis=1)
    targets = df_performance[["Heating load", "Cooling load"]].copy()
    print("{:<16}:\t{} columns {} rows\n{:<16}:\t{} columns {} rows".format("Features", np.shape(features)[1], np.shape(features)[0], "Targets", np.shape(targets)[1], np.shape(targets)[0]))
    print()
    print("{:<16}\n\t{:<8}:\t{:<.2f}\n\t{:<8}:\t{:<.2f}".format("Heating loads", "Minimum", targets['Heating load'].min(), "Maximum", targets['Heating load'].max()))
    print()
    print("{:<16}\n\t{:<8}:\t{:<.2f}\n\t{:<8}:\t{:<.2f}".format("Cooling loads", "Minimum", targets['Cooling load'].min(), "Maximum", targets['Cooling load'].max()))
   
    
    #print("features.to_numpy()", features.to_numpy())
    #print("np.array(features)", np.array(features))
    
    #return features, targets
    return features.to_numpy(), targets.to_numpy()


# np.shape(features[1]) == 8, therefore function 8 coefficients
def num_coefficients_8(d):
    t = 0
    for n in range(d + 1):
        # this needs to be nested 8 times
        for i in range(n + 1):
            for j in range(i + 1):
                for k in range(j + 1):
                    for l in range(k + 1):
                        for m in range(l + 1):
                            for n in range(m + 1):
                                for o in range(n + 1):
                                    if i + j + k + l + m + n + o == n:
                                        t += 1
                                    #for p in range(o + 1):
                                        #if i + j + k + l + m + n + o + p == n:
                                            #t += 1
    return t


def calculate_model_function(deg, features, p):
    result = np.zeros(features.shape[0])
    z = 0
    for n in range(deg + 1):
        # this needs to be nested 8 times
        for i in range(n + 1):
            for j in range(i + 1):
                for k in range(j + 1):
                    for l in range(k + 1):
                        for m in range(l + 1):
                            for n in range(m + 1):
                                for o in range(n + 1):
                                        #result += p[z] * (features[:, 0] ** i) * (features[:, 1] ** (n - i) * (features[:, 2] ** (i - j)) * (features[:, 3] ** (j - k)) * (features[:, 4] ** (k - l)) * (features[:, 5] ** (l - m)) * (features[:, 6] ** (m - n)) * (features[:, 7] ** (n - o)))
                                        result += p[z] * (features[:, 0] ** i) * (features[:, 1] ** (n - i) * (features[:, 2] ** j) * (features[:, 3] ** k) * (features[:, 4] ** l) * (features[:, 5] ** m) * (features[:, 6] ** n) * (features[:, 7] ** o))
                                        z += 1
    return result


def linearize(deg, features, p0):
    f0 = calculate_model_function(deg, features, p0)
    J = np.zeros((len(f0), len(p0)))
    epsilon = 1e-6
    for i in range(len(p0)):
        p0[i] += epsilon
        fi = calculate_model_function(deg, features, p0)
        p0[i] -= epsilon
        di = (fi - f0) / epsilon
        J[:, i] = di
    return f0, J


def calculate_update(y, f0, J):
    l = 1e-2
    N = np.matmul(J.T, J) + l * np.eye(J.shape[1])
    r = y - f0
    n = np.matmul(J.T, r)
    dp = np.linalg.solve(N, n)
    return dp








def main():
    #------------------------------------------------------------------------------------
    #--------------------------- TASK 1 (input data, 6 points) --------------------------
    #------------------------------------------------------------------------------------
    # COMPLETED - Load the data from the file and split it into features [1 point]
    # COMPLETED - and targets [1 point]
    # COMPLETED - Output minimum and maximum heating/cooling [4 points].
    file = "energy_performance.csv"
    features, targets = preprocess_data(file)
    
    max_iter = 10
    for deg in range(3):
        p0 = np.zeros(num_coefficients_8(deg))
        for i in range(max_iter):
            f0, J = linearize(deg, features, p0)
            dp = calculate_update(targets, f0, J)
            print(dp)
            p0 += dp

    #------------------------------------------------------------------------------------
    #--------------------------- TASK 2 (model function, 4 points) --------------------------
    #------------------------------------------------------------------------------------
    #Create a polynomial model function that takes as input parameters the degree of the polynomial, a 
    #list of feature vectors as extracted in task 1, and a parameter vector of coefficients andcalculates 
    #the estimated target vector using a multi-variate polynomial of the specified degree [3 points]. 
    #Create a second function that determines the correct size for the parameter vector from the degree 
    #of the multi-variate polynomial [1 point].


    #------------------------------------------------------------------------------------
    #--------------------------- TASK 3 (linearization, 4 points) --------------------------
    #------------------------------------------------------------------------------------
    #Create a function that calculates the value of the model function implemented in task 2 and its 
    #Jacobian at a given linearization point using the numerical linearisation procedure discussed in the 
    #lectures/labs. The function should take the degree of the polynomial, a list of feature vectors as 
    #extracted in task 1, and the coefficients of the linearization point as input and calculate the 
    #estimated target vector and the Jacobian at the linearization point as output.

    
    #------------------------------------------------------------------------------------
    #--------------------------- TASK 4 (parameter update, 4 points) --------------------------
    #------------------------------------------------------------------------------------
    #Create a function that calculates the optimal parameter update from the training target vector 
    #extracted in task 1 and the estimated target vector and Jacobian calculated in task 3 following the 
    #procedure discussed during the lectures/labs. To do that start with calculating the normal equation 
    #matrix; make sure that you add a regularisation term to prevent the normal equation system from 
    #being singular. Now calculate the residual and built the normal equation system. Solve the normal 
    #equation system to obtain the optimal parameter update. The function should take the training 
    #target vector and the estimated target vector and Jacobian at the linearization point as input and 
    #calculate the optimal parameter update vector as output.
    
    
    #------------------------------------------------------------------------------------
    #--------------------------- TASK 5 (regression, 5 points) --------------------------
    #------------------------------------------------------------------------------------
    #Create a function that calculates the coefficient vector that best fits the training data. To do that, 
    #initialise the parameter vector of coefficients with zeros. Then setup an iterative procedure that 
    #alternates linearization and parameter update following the approach discussed during the 
    #lectures/labs. The function should take the degree of the polynomial, the training data features, and 
    #the training data targets as input and return the best fitting polynomial coefficient vector as output.
    
    
    #------------------------------------------------------------------------------------
    #--------------------------- TASK 6 (model selection, 6 points) --------------------------
    #------------------------------------------------------------------------------------
    #Setup two cross-validation procedures, one for the heat loads and one for cooling loads [1 point].
    #Calculate the difference between the predicted target and the actual target for the test set in each 
    #cross-validation fold [1 point] and output the mean of absolute differences across all folds for both 
    #the heating load estimation as well as the cooling load estimation [2 points]. Using this as a quality 
    #metric, evaluate polynomial degrees ranging between 0 and 2 to determine the optimal degree for 
    #the model function for both the heating as well as the cooling loads [2 points].


    #------------------------------------------------------------------------------------
    #--------------------------- TASK 7 (evaluation and visualisation of results, 6 points) --------------------------
    #------------------------------------------------------------------------------------
    #Now using the full dataset, estimate the model parameters for both the heating loads as well as the 
    #cooling loads using the selected optimal model function as determined in task 6 [1 point]. Calculate 
    #the predicted heating and cooling loads using the estimated model parameters for the entire dataset 
    #[1 point]. Plot the estimated loads against the true loads for both the heating and the cooling case [2
    #points]. Calculate and output the mean absolute difference between estimated heating/cooling 
    #loads and actual heating/cooling loads [2 points].
    
  
main()