# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#import key packageskeras.utils.plot_model(model, "my_first_model.png")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf


import keras.backend as k
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# %% LOAD DATA
df = pd.read_csv("train_data.csv") 
df = 
df.head() #length 1095

# %%
learn_rate = 0.1 #,.01, .003, .001
hidden_neurons = 6 # 3,4,5 
n_batch = 50 #none
act_f = "softmax" #'sigmoid'
reg_param = .001 #.0001, 0


# %% [markdown]


# %% CREATE GOOD DATA & OTHER FUNCTIONS

def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

  def plot_train_history(history, title):
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(len(loss))

  plt.figure()

  plt.plot(epochs, loss, 'b', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title(title)
  plt.legend()

  plt.show()

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


# %%

features_considered = ['drybulb','dewpnt']

for i in range(1,9):
    features_considered += [str(i) ]

features = df[features_considered]
features.index = df['Unnamed: 0']
features.head()

# %% standardize dataset
TRAIN_SPLIT = 1095 #Always 3 years
BATCH_SIZE = 50
BUFFER_SIZE = 10 #What is this?
EPOCHS = 50
EVALUATION_INTERVAL = 200

features.plot(subplots=True)

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std

# %%

past_history = 2
future_target = 1
STEP = 1

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)
print ('Single window of past history : {}'.format(x_train_single[0].shape))

# %% TRAIN VAL DATA

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))

# train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
train_data_single = train_data_single.cache().batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()


# %%

# %% custom loss function
def custom_loss(y_true, y_pred):
    
    # calculate loss, using likehood function for residual Rt

    mean = y_pred[0]
    var = k.square(y_pred[1])
    log_L = 1/k.log( k.sqrt(2*np.pi*var))* k.square(mean-y_true)/(2*var)

    #return -10**3*k.sum(k.log(loss), axis=-2)
    return -(10**3)*k.mean(-log_L)

# %% NAX MODEL

single_step_model = tf.keras.models.Sequential()

act_reg = tf.keras.regularizers.l1 (reg_param)

single_step_model.add(tf.keras.layers.SimpleRNN(6, activation='sigmoid',
                                           input_shape=x_train_single.shape[-2:],
                                           activity_regularizer= act_reg ))
single_step_model.add(tf.keras.layers.Dense(2))
opt = tf.keras.optimizers.Adam(learn_rate)
single_step_model.compile(optimizer=opt, loss=custom_loss)

# %%

for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)

# %%

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)


# %% plot train history

plot_train_history(single_step_history, 'Single Step Training and validation loss')


# %%
