# %% import packages
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from tf_ts_functions import multivariate_data
from custom_loss import custom_loss

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# %% load dataset
df = pd.read_csv("train_data.csv",index_col=0) 
df.head()   # length 1095

features_considered = ['drybulb','dewpnt']

for i in range(1,9):
    features_considered += [str(i) ]

features = df[features_considered]
features.head()

labels=np.array(df['residuals'])
    
features['1'] = (features['1']-features['1'].min())/(features['1'].max()-features['1'].min())

dataset_prec = features.values


# %% define function to standardize dataset

def standardize_dataset(df, TRAIN_SPLIT = 1095):
    
    data_mean = df[:TRAIN_SPLIT].mean(axis=0)
    data_std = df[:TRAIN_SPLIT].std(axis=0)

    std_dataset = (df-data_mean)/data_std

    return(std_dataset)


# %% standardize dataset

dataset = standardize_dataset(dataset_prec)
print(dataset)


# %% function to build x_train and y_train

def build_train(dataset, labels, TRAIN_SPLIT = 1095, past_history = 2, future_target = 0, STEP = 1):
    
    f_x_train, f_y_train = multivariate_data(dataset, labels, 0,
                                            TRAIN_SPLIT, past_history,
                                            future_target, STEP,
                                            single_step=True)
    return f_x_train, f_y_train


# %% build x_train, y_train

x_train, y_train = build_train(dataset, labels)


# %% function to build x_val and y_val

def build_val(dataset, labels, TRAIN_SPLIT = 1095, VAL_SPLIT = 1095+365, past_history = 2, future_target = 0, STEP = 1):
    
    f_x_val, f_y_val = multivariate_data(dataset, labels,
                                        TRAIN_SPLIT, VAL_SPLIT, past_history,
                                        future_target, STEP,
                                        single_step=True)
    return f_x_val, f_y_val


# %% build x_val, y_val

x_val, y_val = build_val(dataset, labels)


# %% function inside for
def for_function(x_train, y_train, HIDDEN_NEURONS, ACT_FUN, LEARN_RATE, 
                BATCH_SIZE, REG_PARAM, OUTPUT_NEURONS = 2, BUFFER_SIZE = 10, 
                LOSS_FUNCTION = custom_loss, EPOCHS = 10, EVALUATION_INTERVAL = 200):
    
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(BATCH_SIZE).repeat()

    # build model
    model = tf.keras.models.Sequential()
    act_reg = tf.keras.regularizers.l1 (REG_PARAM)
    model.add(tf.keras.layers.SimpleRNN(HIDDEN_NEURONS,
                                        input_shape=x_train.shape[-2:],
                                        activation=ACT_FUN,
                                        activity_regularizer= act_reg ))
    model.add(tf.keras.layers.Dense(OUTPUT_NEURONS))
    opt = tf.keras.optimizers.Adam(LEARN_RATE)

    model.compile(optimizer=opt, loss=LOSS_FUNCTION)

    history = model.fit(train_data, epochs=EPOCHS,
                        steps_per_epoch=EVALUATION_INTERVAL,
                        validation_data=val_data,
                        validation_steps=50)

    f_y_pred = model.predict(x_val)

    return f_y_pred


# %% function to compute rmse

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def rmse_NAX(y_pred, df, TRAIN_SPLIT = 1095, past_history = 2, future_target = 0):
    N_val = len(y_pred)

    the_length = len(df['std_demand'])
    START = TRAIN_SPLIT + past_history + future_target
    
    demand_true = pd.Series(df['std_demand'][START:START + N_val])
    demand_NAX = (demand_true - y_val) + y_pred[:,0]

    return rmse(demand_NAX, demand_true)    



# %% search optimal hyper parameters

number_of_neurons_list = [3] #[3, 4, 5, 6]
activation_function_list = ['softmax'] #['softmax', 'sigmoid']
initial_learning_rate_list = [0.1] #[0.1, 0.01, 0.003, 0.001]
batch_size_list = [50] # manca no batch, None non funziona
regularization_parameter_list = [0.001] #[0.001, 0.0001, 0]

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
                    
                    y_pred = for_function(x_train, y_train, number_of_neurons, activation_function,
                             initial_learning_rate, batch_size, regularization_parameter, OUTPUT_NEURONS = 2, 
                             BUFFER_SIZE = 10, LOSS_FUNCTION = custom_loss, EPOCHS = 10, EVALUATION_INTERVAL = 200)
                                         
                    RMSE[n1,n2,n3,n4,n5] = rmse_NAX(y_pred, df)

argmin = np.unravel_index(np.argmin(RMSE,axis=None),RMSE.shape)
hyper_parameters = [number_of_neurons_list[argmin[0]],
                    activation_function_list[argmin[1]], 
                    initial_learning_rate_list[argmin[2]], 
                    batch_size_list[argmin[3]], 
                    regularization_parameter_list[argmin[4]]]

print('\n Hyper parameters')
print(hyper_parameters)
