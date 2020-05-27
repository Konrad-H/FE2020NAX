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


## UNIVARIATE MODEL 
## TO UNDERSTAND THIS 
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
univariate_past_history = 2
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

# %% BASELINE SCENARIO
def baseline(history):
  return np.mean(history)

show_plot([x_train_uni[0], y_train_uni[0], baseline(x_train_uni[0])], 0,
           'Baseline Prediction Example')

# %% RECURRENT NEURAL NETWORK
BATCH_SIZE = 50
BUFFER_SIZE = 10

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_univariate = tf.data.Dataset.from_tensor_slices((x_val_uni, y_val_uni))
val_univariate = val_univariate.batch(BATCH_SIZE).repeat()

# %% COMPILE SIMPLE SIMPLE MODEL
simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')

# %%

for x, y in val_univariate.take(1):
    print(simple_lstm_model.predict(x).shape)

# %% 
EVALUATION_INTERVAL = 200
EPOCHS = 10

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=val_univariate, validation_steps=50)

# %% ADD
for x, y in val_univariate.take(3):
  plot = show_plot([x[0].numpy(), y[0].numpy(),
                    simple_lstm_model.predict(x)[0]], 0, 'Simple LSTM model')
  plot.show()

# %% 

# %% [markdown]


# %% 

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

features_considered = ['drybulb','dewpnt']

for i in range(1,9):
    features_considered += [str(i) ]

features = df[features_considered]
features.index = df['Unnamed: 0']
features.head()

# %% standardize dataset
features.plot(subplots=True)

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std

# %%

past_history = 10
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
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

# %%

single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

# %%

for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)

# %%

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)


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


