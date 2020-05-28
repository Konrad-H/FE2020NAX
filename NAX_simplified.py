# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
#import key packageskeras.utils.plot_model(model, "my_first_model.png")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import tensorflow.keras.backend as k
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from tf_ts_functions import  plot_train_history, multivariate_data


mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False


# %% LOAD DATA
df = pd.read_csv("train_data.csv",index_col=0) 
df.head() #length 1095
## UNIVARIATE MODEL 
## TO UNDERSTAND THIS 
# %% functions


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
def custom_loss(y_true, y_pred):
    
    # calculate loss, using likehood function for residual Rt

    mean = y_pred[0]
    var = y_pred[1]

    log_L = -k.log(2*np.pi*var)/2-k.square(mean-y_true)/(2*var)
    return -(10**3)*k.mean(log_L)

# %% PLOT AND STANDARDIZE
TRAIN_SPLIT = 1095
VAL_SPLIT = 1095+365
BATCH_SIZE = 50
BUFFER_SIZE = 10
EVALUATION_INTERVAL = 200
EPOCHS = 10
HIDDEN_NEURONS=32
REG_PARAM = 0.1
ACT_FUN = 'softmax' #'SIGMOID'
LEARN_RATE = 0.001

tf.random.set_seed(14)


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
                                               TRAIN_SPLIT, VAL_SPLIT, past_history,
                                               future_target, STEP,
                                               single_step=True)
print ('Single window of past history : {}'.format(x_train_single[0].shape))

# %% TRAIN VAL DATA
train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

# %%

act_reg = tf.keras.regularizers.l1 (REG_PARAM)
single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.SimpleRNN(HIDDEN_NEURONS,
                                           input_shape=x_train_single.shape[-2:],
                                           activation=ACT_FUN,
                                           activity_regularizer= act_reg ))
                                           
single_step_model.add(tf.keras.layers.Dense(1))


opt = tf.keras.optimizers.Adam(LEARN_RATE)
# opt = tf.keras.optimizers.RMSprop()
single_step_model.compile(optimizer=opt, loss='mae')

# %%

for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)

# %%

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)

# %% 


y_pred =single_step_model.predict(x_val_single)
N_val = len(y_pred)
the_length = len(df['std_demand'])
START = TRAIN_SPLIT+2 

demand_true = pd.Series(df['std_demand'][START:START+N_val])
demand_NAX = (demand_true- y_val_single)+y_pred[:,0]
demand_GLM = (demand_true- y_val_single)
# %%
plt.figure()
demand_true.plot()
demand_NAX.plot()
demand_GLM.plot()
plt.show()

RMSE_GLM=rmse(demand_GLM, demand_true)
RMSE_NAX=rmse(demand_NAX, demand_true)
print('RMSE_GLM',RMSE_GLM)
print('RMSE_NAX',RMSE_NAX)

# %%
