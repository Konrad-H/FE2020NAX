# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#import key packageskeras.utils.plot_model(model, "my_first_model.png")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from tf_ts_functions import  plot_train_history, multivariate_data
from NAX_functions import custom_loss, inverse_std
from tensorflow.keras.callbacks import EarlyStopping

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# %% LOAD DATA
df = pd.read_csv("train_data.csv",index_col=0) 
df.head() #length 1095
## UNIVARIATE MODEL 
## TO UNDERSTAND THIS 
# %% functions




# %% PLOT AND STANDARDIZE
# RMSE_NAX 15776.347510314612 with sigmoid

START_SPLIT = 0
TRAIN_SPLIT = 1095
VAL_SPLIT = 1095+365
BATCH_SIZE = 50 #None
BUFFER_SIZE = 5

EVALUATION_INTERVAL = 500
EPOCHS = 50 #200
REG_PARAM = 0.0001
ACT_FUN = 'softmax' #'sigmoid' 'softmax'
LEARN_RATE = 0.003
HIDDEN_NEURONS=3 #3
LOSS_FUNCTION =  custom_loss #custom_loss #'mae', 'mse'
OUTPUT_NEURONS= 2 #2
INPUT_SHAPE = (2,10)
STOPPATIENCE = 10

past_history = 2
future_target = 0
STEP = 1
# opt=tf.keras.optimizers.RMSprop()

# %%

def aggregate_data(features,labels,
            START_SPLIT = 0,
            TRAIN_SPLIT = 1095,
            VAL_SPLIT = 1095+365,
            past_history = 2,
            future_target = 0,
            STEP = 1):



    x_train, y_train = multivariate_data(features, labels, START_SPLIT,
                                                    TRAIN_SPLIT, past_history,
                                                    future_target, STEP,
                                                    single_step=True)
    x_val, y_val = multivariate_data(features, labels,
                                                TRAIN_SPLIT, VAL_SPLIT, past_history,
                                                future_target, STEP,
                                                single_step=True)
    #print ('Single window of past history : {}'.format(x_train[0].shape))
    return x_train, y_train,x_val, y_val

def ready_data(x_train, y_train,x_val, y_val,
                BATCH_SIZE = 50,
                shuffle=False,
                BUFFER_SIZE = 50):           
    # %% TRAIN VAL DATA
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if shuffle ==True:
        train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    else:
        train_data = train_data.cache().batch(BATCH_SIZE).repeat()

    val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_data = val_data.batch(BATCH_SIZE).repeat()
    return train_data,val_data
# %%

def NAX_model(INPUT_SHAPE=(2,10),
            REG_PARAM = 0.0001,
            ACT_FUN = 'softmax',
            LEARN_RATE = 0.003,
            HIDDEN_NEURONS=3 ,
            OUTPUT_NEURONS= 2,
            LOSS_FUNCTION =  custom_loss):

    act_reg = tf.keras.regularizers.l1 (REG_PARAM)
    opt = tf.keras.optimizers.Adam(LEARN_RATE)

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.SimpleRNN(HIDDEN_NEURONS,
                                            input_shape=INPUT_SHAPE,
                                            activation=ACT_FUN,
                                            activity_regularizer= act_reg ,
                                            #  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1),
                                            # bias_initializer=tf.keras.initializers.Ones()

                                            ))
                                            
    model.add(tf.keras.layers.Dense(OUTPUT_NEURONS))



    # opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss=LOSS_FUNCTION)
    return model

# %%


def demands(y_pred, y_val,df,START):
    N_val = len(y_pred)
    the_length = len(df['std_demand'])

    std_demand_true = pd.Series(df['std_demand'][START:START+N_val])
    std_demand_NAX = (std_demand_true- y_val)+y_pred[:,0]
    std_demand_GLM = (std_demand_true- y_val)
    demand_log_train= np.exp( pd.Series(df['log_demand'][START_SPLIT:TRAIN_SPLIT]) )

    demand_true=inverse_std(std_demand_true,demand_log_train) 
    demand_NAX=inverse_std(std_demand_NAX,demand_log_train) 
    demand_GLM=inverse_std(std_demand_GLM,demand_log_train) 
    return demand_true, demand_NAX, demand_GLM 
