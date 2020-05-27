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

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# %% LOAD DATA
df = pd.read_csv("train_data.csv") 
df.head() #length 1095

# %% functions
def univariate_data(dataset, start_index, end_index, history_size, target_size):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i)
    # Reshape data from (history_size,) to (history_size, 1)
    data.append(np.reshape(dataset[indices], (history_size, 1)))
    labels.append(dataset[i+target_size])
  return np.array(data), np.array(labels)


# %% PLOT AND STANDARDIZE
TRAIN_SPLIT = 1095

tf.random.set_seed(13)

uni_data = df['residuals']
uni_data.index = df['Unnamed: 0']
uni_data.head()

uni_data.plot(subplots=True)

uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
uni_train_std = uni_data[:TRAIN_SPLIT].std()
uni_data = np.array(uni_data-uni_train_mean)/uni_train_std

# %%
univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)
x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

# %%
print ('Single window of past history')
print (x_train_uni[0])
print ('\n Target temperature to predict')
print (y_train_uni[0])

# %% Create Time Steps

def create_time_steps(length):
  return list(range(-length, 0))

def show_plot(plot_data, delta, title):
  labels = ['History', 'True Future', 'Model Prediction']
  marker = ['.-', 'rx', 'go']
  time_steps = create_time_steps(plot_data[0].shape[0])
  if delta:
    future = delta
  else:
    future = 0

  plt.title(title)
  for i, x in enumerate(plot_data):
    if i:
      plt.plot(future, plot_data[i], marker[i], markersize=10,
               label=labels[i])
    else:
      plt.plot(time_steps, plot_data[i].flatten(), marker[i], label=labels[i])
  plt.legend()
  plt.xlim([time_steps[0], (future+5)*2])
  plt.xlabel('Time-Step')
  return plt

show_plot([x_train_uni[0], y_train_uni[0]], 0, 'Sample Example')

# %% 
def baseline(history):
  return np.mean(history)

show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
           'Baseline Prediction Example')

# %% custom loss function
def custom_loss(y_true, y_pred):
    
    # calculate loss, using likehood function for residual Rt

    mean = y_pred[0]
    var = (y_pred[1])**2 
    loss = np.exp(-(y_true-mean)**2 /(2*var) ) / (2*np.pi*var)**.5
        
    return -10**3*tf.reduce_mean(np.log(loss), axis=-1)


# %%
#optimizer 
learn_rate = 0.1 #,.01, .003, .001
hidden_neurons = 6 # 3,4,5 
n_batch = 50 #none
act_f = "softmax" #'sigmoid'
reg_param = .001 #.0001, 0


# %%
data.head()


# %%
input_size = 10
output_states=2
look_back = 1


# %%
X = [data['drybulb'], data['dewpnt'], t, np.sin(omega*t), np.cos(omega*t), np.sin(2*omega*t), np.cos(2*omega*t),D_weekend.Sat, D_weekend.Sun, D_holiday.iloc[:,1]]


x_train= np.transpose(X)

len((x_train))


# %%
y_train = np.array(data['demand'])


# %%
model = keras.models.Sequential()

# NOT USED
act_reg = keras.regularizers.l1 (reg_param)


# %%
model.add(layers.LSTM ( 5 , input_shape=(input_size, look_back), return_sequences=True ))


# %%
# SIMPLE LSTM MODEL
model.add(layers.Dense(output_states))

# Optimizer
opt = keras.optimizers.Adam(learning_rate=learn_rate)
model.compile(loss='mae', optimizer=opt)


# %%

# FIT
model.fit(x_train, y_train, epochs=150, batch_size=n_batch , verbose=1)


# %%


