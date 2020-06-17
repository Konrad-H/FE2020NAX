# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series
from MLE_loss import loss_strike
from tensorflow.keras.callbacks import EarlyStopping
from standard_and_error_functions import destd

# %% USED PARAMETERS IN THE CODE

# How to split the data
START_SPLIT = 0       #FIRST LINE
TRAIN_SPLIT = 1095    #SPLIT LINE
END_SPLIT = 1095+365  #LAST LINE 

# BATCH SIZE to be used in the training
BATCH_SIZE = 50 #None

# NUMBER OF EPOCHS for the training
EPOCHS = 500 #200

# PARAMETERS OF THE MODEL
REG_PARAM = 0.0001  # L1 normalization parameter
ACT_FUN = 'softmax' # 'sigmoid' 'softmax'
LEARN_RATE = 0.003  # learn rate of ADAM function
HIDDEN_NEURONS = 3  # as written
LOSS_FUNCTION, _ =  loss_strike(.0001) # as written
OUTPUT_NEURONS= 2             # as written
OUT_KERNEL = 'glorot_uniform'     # weights for the Kernel Init
OUT_BIAS = 'zeros'            # weights for the BIAS Init

# Shape of the input of the model, 2 time steps, 10 features
INPUT_SHAPE = (2,10)

# How many iterations before keras callbacks stops the function
STOPPATIENCE = 50

# Parameters of Multivariate Data
past_history = 1  #Number of steps back in time, taken for prediction
future_target = 0 #How much to predict in advance


# %%

def prep_data(df, START_SPLIT = 0,
                    TRAIN_SPLIT = 1095,
                    END_SPLIT = 1095+365):

    # Takes a df and outputs the needed features for our model, standardized
    #
    # INPUTS:
    # df:           pandas dataframe with the features and labels
    #
    # OUTPUTS:
    # dataset:      numpy array with the features of our dataset
    # labels:       numpy array with labels of out dataset

    features_considered = ['drybulb','dewpnt']

    for i in range(1,9):
        features_considered += [str(i)]

    features = df[features_considered]
    features.head()
    # features.plot(subplots=True)

    labels = np.array(df['residuals'])

    # %% standardize features s.t. mean = 0, variance = 1
    dataset = features.values
    data_mean = dataset[START_SPLIT:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[START_SPLIT:TRAIN_SPLIT].std(axis=0)

    dataset = (dataset-data_mean)/data_std
    
    return dataset, labels


def multivariate_data(features, target, start_index, end_index, history_size,
                      target_size, single_step = False):

  # Uses multivariate data to aggregate the data form features and target
  # in couples in the structure (x(t-1),x(t)) (y(t)) over the time window 
  # given by start_index and end_index
  #
  # Inputs and Outputs as in aggregate_data
  
  data = []
  labels = []
  start_index = start_index + history_size
  if end_index is None:
    end_index = len(features) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i+1)
    data.append(features[indices])

    if single_step:
      labels.append(target[i])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)


def aggregate_data(features, labels,
            START_SPLIT = 0,
            TRAIN_SPLIT = 1095,
            END_SPLIT = 1095+365,
            past_history = 1,
            future_target = 0):
          
    # Uses multivariate data to aggregate the data form features and labels
    # in couples in the structure (x(t-1),x(t)) (y(t)), both for train and
    # validation set
    #
    # INPUTS:
    # features, labels: output from prep_data
    # others previously explained
    #
    # OUTPUTS:
    # x's tensor of dim ?,past_history, length(features), containing input of the Neural Network
    # y's tensor of dim ?,1, length(labels), contatining labels of the Neural Network
    # where ? is the dim of train or validation respectively

    x_train, y_train = multivariate_data(features, labels, START_SPLIT,
                                                    TRAIN_SPLIT, past_history,
                                                    future_target,
                                                    single_step=True)
    x_val, y_val = multivariate_data(features, labels,
                                                TRAIN_SPLIT, END_SPLIT, past_history,
                                                future_target,
                                                single_step=True)
    #print ('Single window of past history : {}'.format(x_train[0].shape))
    return x_train, y_train,x_val, y_val


def NAX_model(INPUT_SHAPE=(2,10),
            REG_PARAM = 0.0001,
            ACT_FUN = 'softmax',
            LEARN_RATE = 0.003,
            HIDDEN_NEURONS = [3] ,
            OUTPUT_NEURONS = 2,
            LOSS_FUNCTION = 'mse',
            OUT_KERNEL = 'glorot_uniform'    ,
            OUT_BIAS = 'zeros',
            HID_KERNEL = 'glorot_uniform'    ,
            HID_REC = "orthogonal",
            HID_BIAS = 'zeros',
            N_LAYERS = 1
            ):
    
    # Using the parameters previously defined returns the NAX model,
    # composed by a simple RNN layer and a Dense layer as output

    act_reg = tf.keras.regularizers.l1(REG_PARAM)
    opt = tf.keras.optimizers.Adam(learning_rate=LEARN_RATE)
    
    model = tf.keras.models.Sequential()
    layers = len(HIDDEN_NEURONS)
    inputs = [INPUT_SHAPE]
    for i in range(layers-1):
      inputs += [(INPUT_SHAPE[0],HIDDEN_NEURONS[i])]
    return_sequences = True
    for i in range(layers):
      layer_input = inputs[i]
      layer_neurons= HIDDEN_NEURONS[i]
      if i==layers-1:
        return_sequences = False
      model.add(tf.keras.layers.SimpleRNN(layer_neurons,
                                          input_shape = layer_input,
                                          activation = ACT_FUN,
                                          activity_regularizer = act_reg,
                                          kernel_initializer=HID_KERNEL,
                                          recurrent_initializer=HID_REC,
                                          bias_initializer=HID_BIAS,
                                          return_sequences = return_sequences
                                          ))
                          
    model.add(tf.keras.layers.Dense(OUTPUT_NEURONS,
              kernel_initializer=OUT_KERNEL,
              bias_initializer=OUT_BIAS
              ))
    
    model.compile(optimizer=opt, loss=LOSS_FUNCTION)
    return model


def demands(y_pred, y_val, df, START, M, m):
    
    # Calculates the demands in original units from the standard predictions
    #
    # INPUTS:
    # y_pred: 2-dim array with predictions and 'raw' std dev (needs y2var)
    # y_val : 1-dim array with true values
    # df    : Original dataframe used
    # START : First predicted position in df
    # M     : Maximum log_demand in 2008 - 2016
    # m     : Minimum log_demand in 2008 - 2016
    #
    # OUTPUTS:
    # demand_true: True original demand
    # demand_NAX:  NAX predicited demand
    # demand_GLM:  GLM predicted demand

    N_val = len(y_pred)

    std_demand_true = Series(df['std_demand'][START:START+N_val])
    std_demand_GLM = (std_demand_true- y_val)
    std_demand_NAX = std_demand_GLM+y_pred[:,0]

    demand_true = destd(std_demand_true, M, m)
    demand_NAX = destd(std_demand_NAX, M, m)
    demand_GLM = destd(std_demand_GLM, M, m)
    return demand_true, demand_NAX, demand_GLM 


def plot_train_history(history, title):
    
    # Plots the val and train loss across epochs
    # 
    # INPUTS:
    # history: output from model.fit
    # title:   name of plot

  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()


def one_NAX_iteration(df,START_SPLIT = 0,
                TRAIN_SPLIT = 1095,
                END_SPLIT = 1095+365,
                BATCH_SIZE = 50,
                EPOCHS = 500,
                REG_PARAM = 0.0001,
                ACT_FUN = 'softmax',
                LEARN_RATE = 0.003,
                HIDDEN_NEURONS = 3 ,
                LOSS_FUNCTION = 'mse',
                OUTPUT_NEURONS= 2,
                STOPPATIENCE = 50,
                past_history = 1,
                future_target = 0,
                VERBOSE = 1,
                VERBOSE_EARLY = 1,
                OUT_KERNEL = 'glorot_uniform',
                OUT_BIAS = 'zeros',
                HID_KERNEL = 'glorot_uniform',
                HID_REC = "orthogonal",
                HID_BIAS = 'zeros',
                N_LAYERS = 1):
    
    # Takes the previously defined hyperparameters, implement the NAX model
    # and returns the prediction on the validation set and the fitted model
    #
    # OUTPUTS:
    # y_pred:  predictions of the model for the test set
    # history: all the information stored in the model fitting
    # model:   the model which we trained

    # Neural Network features and labels are defined
    features, labels = prep_data(df,
                        START_SPLIT = START_SPLIT,
                        TRAIN_SPLIT = TRAIN_SPLIT,
                        END_SPLIT = END_SPLIT)
    
    # Inputs and targets both on train and validation test are defined
    x_train, y_train,x_val, y_val = aggregate_data(features,labels,
                                    START_SPLIT = START_SPLIT,
                                    TRAIN_SPLIT = TRAIN_SPLIT,
                                    END_SPLIT = END_SPLIT,
                                    past_history = past_history,
                                    future_target = future_target)

    # The NAX Model is built
    model = NAX_model(INPUT_SHAPE = x_train.shape[-2:],
                REG_PARAM = REG_PARAM,
                ACT_FUN = ACT_FUN,
                LEARN_RATE = LEARN_RATE,
                HIDDEN_NEURONS = HIDDEN_NEURONS ,
                OUTPUT_NEURONS = OUTPUT_NEURONS,
                LOSS_FUNCTION =  LOSS_FUNCTION,
                OUT_KERNEL = OUT_KERNEL,
                OUT_BIAS = OUT_BIAS,
                HID_KERNEL = HID_KERNEL,
                HID_REC = HID_REC,
                HID_BIAS = HID_BIAS,
                N_LAYERS = N_LAYERS
                )

    # EarlyStopping callbacks option is defined
    EARLYSTOP = EarlyStopping(monitor='val_loss', mode='min', verbose=VERBOSE_EARLY, patience=STOPPATIENCE)

    # NAX model is calibrated
    history=model.fit(
        x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, callbacks=[EARLYSTOP],
        validation_data=(x_val,y_val), validation_batch_size=BATCH_SIZE)
    
    # Prediction on validation set
    y_pred = model.predict(x_val)

    return y_pred, history, model
