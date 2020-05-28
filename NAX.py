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
from custom_loss import custom_loss

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


# %% PLOT AND STANDARDIZE
START_SPLIT = 0
TRAIN_SPLIT = 1095
VAL_SPLIT = 1095+365
BATCH_SIZE = 50 #None
BUFFER_SIZE = 10

EVALUATION_INTERVAL = 1093
EPOCHS = 20
REG_PARAM = 0.0001
ACT_FUN = 'softmax' #'sigmoid' 'softmax'
LEARN_RATE = 0.003
HIDDEN_NEURONS=3
LOSS_FUNCTION =  custom_loss #custom_loss #'mae', 'mse'
OUTPUT_NEURONS= 2 #2


opt = tf.keras.optimizers.Adam(LEARN_RATE)
opt=tf.keras.optimizers.RMSprop()
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
data_mean = dataset[START_SPLIT:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[START_SPLIT:TRAIN_SPLIT].std(axis=0)

dataset = (dataset-data_mean)/data_std

# %%

past_history = 2
future_target = 0
STEP = 1

x_train, y_train = multivariate_data(dataset, labels, START_SPLIT,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val, y_val = multivariate_data(dataset, labels,
                                               TRAIN_SPLIT, VAL_SPLIT, past_history,
                                               future_target, STEP,
                                               single_step=True)
print ('Single window of past history : {}'.format(x_train[0].shape))

# %% TRAIN VAL DATA
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()

# %%
model = tf.keras.models.Sequential()

act_reg = tf.keras.regularizers.l1 (REG_PARAM)

model.add(tf.keras.layers.SimpleRNN(HIDDEN_NEURONS,
                                           input_shape=x_train.shape[-2:],
                                           activation=ACT_FUN,
                                           activity_regularizer= act_reg ))
                                           
model.add(tf.keras.layers.Dense(OUTPUT_NEURONS))



# opt = tf.keras.optimizers.RMSprop()
model.compile(optimizer=opt, loss=LOSS_FUNCTION)

# %%

for x, y in val_data.take(1):
  print(model.predict(x).shape)

# %%

history = model.fit(train_data, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data,
                                            validation_steps=50)
plot_train_history(history,"Loss of model")
# %%
y_pred =model.predict(x_val)
N_val = len(y_pred)
the_length = len(df['std_demand'])
START = TRAIN_SPLIT+past_history+future_target

demand_true = pd.Series(df['std_demand'][START:START+N_val])
demand_NAX = (demand_true- y_val)+y_pred[:,0]
demand_GLM = (demand_true- y_val)
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
