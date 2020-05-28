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
df = df[:4*365]
df.head() #length 1095

# %% PLOT DATA
# df.plot(subplots=True)
# %%
learn_rate = 0.1 #,.01, .003, .001
hidden_neurons = 6 # 3,4,5 
n_batch = 50 #none
act_f = "softmax" #'sigmoid'
reg_param = .001 #.0001, 0

TRAIN_SPLIT = 1095 #Always 3 years
BATCH_SIZE = 50
BUFFER_SIZE = 15 #What is this?
EPOCHS = 10
EVALUATION_INTERVAL = 200


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

# %%

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


# %%
labels_considered = ['residuals']
features_considered = ['drybulb','dewpnt']

for i in range(1,9):
    features_considered += [str(i) ]

dataset = df[labels_considered+features_considered]
dataset.index = df['Unnamed: 0']
dataset.head() 


# %% standardize dataset (already standardized)
 
dataset['1'] = dataset['1']/np.max(dataset['1'])

features = np.array(dataset)[:,1:]
target = np.array(dataset)[:,0]






# %%

past_history = 2
future_target = 1
STEP = 1

x_train, y_train = multivariate_data(features, target, 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val, y_val = multivariate_data(features, target,
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)
print ('Single window of past history : {}'.format(x_train[0].shape))

# %% TRAIN VAL DATA

train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))

#train_data = train_data.cache().batch(BATCH_SIZE).repeat()
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()


val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()


# %%

# %% custom loss function
def custom_loss(y_true, y_pred):
    
    # calculate loss, using likehood function for residual Rt

    mean = y_pred[0]
    var = k.square(y_pred[1])
    #log_L = -k.log( k.sqrt(2*np.pi*var))* k.square(mean-y_true)/(2*var)
    #L = k.exp(k.square(mean-y_true)/(2*var))/ k.sqrt(2*np.pi*var)
    #return -(10**3)*k.mean(k.log(L))

    log_L = -k.log(2*np.pi*var)/2-k.square(mean-y_true)/(2*var)
    return -(10**3)*k.mean(log_L)

# %% NAX MODEL

NAX_model = tf.keras.models.Sequential()

act_reg = tf.keras.regularizers.l1 (reg_param)

NAX_model.add(tf.keras.layers.SimpleRNN(hidden_neurons, activation=act_f ,
                                           input_shape=x_train.shape[-2:],
                                           activity_regularizer= act_reg ))
NAX_model.add(tf.keras.layers.Dense(2))
opt = tf.keras.optimizers.Adam(learn_rate)
NAX_model.compile(optimizer=opt, loss=custom_loss)

# %%

for x, y in val_data.take(1):
  print(NAX_model.predict(x).shape)

# %%

NAX_history = NAX_model.fit(train_data, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data,
                                            validation_steps=50)


# %% plot train history

plot_train_history(NAX_history, 'Single Step Training and validation loss')


# %%

NAX_model.predict(x_val)



# %%
