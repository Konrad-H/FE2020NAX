import numpy as np
import pandas as pd 
import os
import sys

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# y_true = df['demand']

number_of_neurons_list = [3, 4, 5, 6]
activation_function_list = ['softmax', 'sigmoid']
initial_learning_rate_list = [0.1, 0.01, 0.003, 0.001]
batch_size_list = [50, 'no batch'] # 'no batch' = grandezza del dataset
regularization_parameter_list = [0.001, 0.0001, 0]

RMSE = np.zeros((len(number_of_neurons_list), len(activation_function_list), len(initial_learning_rate_list), len(batch_size_list), len(regularization_parameter_list)))

for n1 in range(len(number_of_neurons_list)):
    for n2 in range(len(activation_function_list)):
        for n3 in range(len(initial_learning_rate_list)):
            for n4 in range(len(batch_size_list)):
                for n5 in range(len(regularization_parameter_list)):
                    number_of_neurons = number_of_neurons_list[n1]
                    activation_function = activation_function_list[n2]
                    initial_learning_rate = initial_learning_rate_list[n3]
                    batch_size = batch_size_list[n4]
                    regularization_parameter = regularization_parameter_list[n5]
                    # [mu, sigma] 
                    # 
                    #RMSE[n1,n2,n3,n4,n5] = rmse(predictions, targets)

# 
argmin = np.unravel_index(np.argmin(RMSE,axis=None),RMSE.shape)
hyper_parameters = [number_of_neurons_list[argmin[0]], activation_function_list[argmin[1]], initial_learning_rate_list[argmin[2]], batch_size_list[argmin[3]], regularization_parameter_list[argmin[4]]]
