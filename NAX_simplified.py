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

from tf_ts_functions import univariate_data, show_plot, plot_train_history, multivariate_data


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# %% LOAD DATA
df = pd.read_csv("train_data.csv",index_col=0) 

first_year = 0
last_year = 365*4
df=df[first_year:last_year]
df.head() #length 1095
## UNIVARIATE MODEL 
## TO UNDERSTAND THIS 
# %% functions


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
# %% PLOT AND STANDARDIZE
TRAIN_SPLIT = 1095

tf.random.set_seed(13)

uni_data = df['residuals']

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

# %%

features_considered = ['drybulb','dewpnt']

for i in range(1,9):
    features_considered += [str(i) ]

features = df[features_considered]
features.head()

labels=np.array(df['residuals'])
# %% standardize dataset
#features.plot(subplots=True)
features['1'] = (features['1']-features['1'].min())/(features['1'].max()-features['1'].min())

dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std

# %%

past_history = 2
future_target = 0
STEP = 1

x_train_single, y_train_single = multivariate_data(dataset, labels, 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, labels,
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
                                            steps_per_epoch=EVALUATION_INTERVAL),
                                            validation_data=val_data_single,
                                            validation_steps=50)

# %% 


y_pred =single_step_model.predict(x_val_single)
N_val = len(y_pred)


demand_true = pd.Series(df['std_demand'][-N_val:])
demand_NAX = (demand_true- y_val_single)+y_pred[:,0]
demand_GLM = (demand_true- y_val_single)
# %%
demand_true.plot()
demand_NAX.plot()
demand_GLM.plot()
RMSE_GLM=rmse(demand_GLM, demand_true)
RMSE_NAX=rmse(demand_NAX, demand_true)
print('RMSE_GLM',RMSE_GLM)
print('RMSE_NAX',RMSE_NAX)

# %%

# %%
