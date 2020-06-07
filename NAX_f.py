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
from NAX_functions import custom_loss
from tensorflow.keras.callbacks import EarlyStopping
from dest_f import destd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# %%
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
STOPPATIENCE = 50

past_history = 2
future_target = 0
STEP = 1
# opt=tf.keras.optimizers.RMSprop()

# %%

def prep_data(df, START_SPLIT = 0,
                    TRAIN_SPLIT = 1095,
                    VAL_SPLIT = 1095+365,
                    DAYS_NOT_STD = False):
    features_considered = ['drybulb','dewpnt']

    for i in range(1,9):
        features_considered += [str(i)]

    features = df[features_considered]
    features.head()

    labels = np.array(df['residuals'])
    # %% standardize dataset
    # features.plot(subplots=True)
    if DAYS_NOT_STD:
        features['1'] = (features['1']-features['1'].min())/(features['1'].max()-features['1'].min())
        
    dataset = features.values
    data_mean = dataset[START_SPLIT:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[START_SPLIT:TRAIN_SPLIT].std(axis=0)
    
    dataset = (dataset-data_mean)/data_std
    
    return dataset, labels


# %%

def aggregate_data(features, labels,
            START_SPLIT = 0,
            TRAIN_SPLIT = 1095,
            VAL_SPLIT = 1095+365,
            past_history = 1,
            future_target = 0):

    x_train, y_train = multivariate_data(features, labels, START_SPLIT,
                                                    TRAIN_SPLIT, past_history,
                                                    future_target,
                                                    single_step=True)
    x_val, y_val = multivariate_data(features, labels,
                                                TRAIN_SPLIT, VAL_SPLIT, past_history,
                                                future_target,
                                                single_step=True)
    #print ('Single window of past history : {}'.format(x_train[0].shape))
    return x_train, y_train,x_val, y_val


# %%

def NAX_model(INPUT_SHAPE=(2,10),
            REG_PARAM = 0.0001,
            ACT_FUN = 'softmax',
            LEARN_RATE = 0.003,
            HIDDEN_NEURONS = 3 ,
            OUTPUT_NEURONS = 2,
            LOSS_FUNCTION = custom_loss):

    act_reg = tf.keras.regularizers.l1(REG_PARAM)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
    
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.SimpleRNN(HIDDEN_NEURONS,
                                        input_shape = INPUT_SHAPE,
                                        activation = ACT_FUN,
                                        activity_regularizer = act_reg,
                                        kernel_initializer=tf.keras.initializers.he_normal(),
                                        bias_initializer=tf.keras.initializers.he_uniform()))
    
    kernel = np.array([[ 0.4, -0.2],
                       [-0.8, -0.1],
                       [ 0.5, -0.2]])
    #kernel = np.array([[ 0.44, -0.15],
    #                   [-0.81, -0.06],
    #                   [ 0.46, -0.16]])
    bias = np.array([0.2, 0.1])
    #bias = np.array([0.22, 0.11])                                   
    model.add(tf.keras.layers.Dense(OUTPUT_NEURONS,
              kernel_initializer=tf.keras.initializers.Constant(kernel),
              bias_initializer=tf.keras.initializers.Constant(bias)))
    
    # opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss=LOSS_FUNCTION)
    return model

# %%

def demands(y_pred, y_val, df, START, M, m): #df_NAX solo 4 anni
    N_val = len(y_pred)
    
    std_demand_true = pd.Series(df['std_demand'][START:START+N_val])
    std_demand_GLM = (std_demand_true- y_val)
    std_demand_NAX = std_demand_GLM+y_pred[:,0]
    #demand_log_train= pd.Series(df['log_demand']) 

    demand_true = destd(std_demand_true, M, m)
    demand_NAX = destd(std_demand_NAX, M, m)
    demand_GLM = destd(std_demand_GLM, M, m)
    return demand_true, demand_NAX, demand_GLM 


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


from NAX_f import prep_data, aggregate_data, NAX_model, demands


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# %%

def one_NAX_iteration(df,START_SPLIT = 0,
                TRAIN_SPLIT = 1095,
                VAL_SPLIT = 1095+365,
                BATCH_SIZE = 50,
                EPOCHS = 500,
                REG_PARAM = 0.0001,
                ACT_FUN = 'softmax',
                LEARN_RATE = 0.003,
                HIDDEN_NEURONS = 3 ,
                LOSS_FUNCTION = custom_loss,
                OUTPUT_NEURONS= 2,
                STOPPATIENCE = 50,
                past_history = 1,
                future_target = 0,
                VERBOSE = 1,
                VERBOSE_EARLY = 1):

    features, labels = prep_data(df,
                        START_SPLIT = START_SPLIT,
                        TRAIN_SPLIT = TRAIN_SPLIT,
                        VAL_SPLIT = VAL_SPLIT, DAYS_NOT_STD=True)
    
    x_train, y_train,x_val, y_val = aggregate_data(features,labels,
                                    START_SPLIT = START_SPLIT,
                                    TRAIN_SPLIT = TRAIN_SPLIT,
                                    VAL_SPLIT = VAL_SPLIT,
                                    past_history = past_history,
                                    future_target = future_target)

    # %%

    model = NAX_model(INPUT_SHAPE = x_train.shape[-2:],
                REG_PARAM = REG_PARAM,
                ACT_FUN = ACT_FUN,
                LEARN_RATE = LEARN_RATE,
                HIDDEN_NEURONS = HIDDEN_NEURONS ,
                OUTPUT_NEURONS = OUTPUT_NEURONS,
                LOSS_FUNCTION =  LOSS_FUNCTION)

    EARLYSTOP = EarlyStopping(monitor='val_loss', mode='min', verbose=VERBOSE_EARLY, patience=STOPPATIENCE)
    history=model.fit(
        x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, callbacks=[EARLYSTOP],
        validation_data=(x_val,y_val), validation_batch_size=BATCH_SIZE)
    
    # %%
    y_pred = model.predict(x_val)
    return y_pred, history, model

