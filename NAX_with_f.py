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

from NAX_f import aggregate_data, NAX_model, demands


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
# RMSE_NAX 15776.347510314612 with sigmoid

START_SPLIT = 0
TRAIN_SPLIT = 1095
VAL_SPLIT = 1095+365
BATCH_SIZE = 50 #None
BUFFER_SIZE = 5

EVALUATION_INTERVAL = 500
EPOCHS = 300 #200
REG_PARAM = 0.0001
ACT_FUN = 'softmax' #'sigmoid' 'softmax'
LEARN_RATE = 0.003
HIDDEN_NEURONS=3 #3
LOSS_FUNCTION =  custom_loss #custom_loss #'mae', 'mse'
OUTPUT_NEURONS= 2 #2
STOPPATIENCE = 10

past_history = 2
future_target = 0
STEP = 1

# opt=tf.keras.optimizers.RMSprop()
OPT = tf.keras.optimizers.Adam(LEARN_RATE)
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

x_train, y_train,x_val, y_val = aggregate_data(dataset,labels,
                                  START_SPLIT = START_SPLIT,
                                  TRAIN_SPLIT = TRAIN_SPLIT,
                                  VAL_SPLIT = VAL_SPLIT,
                                  past_history = past_history,
                                  future_target = future_target,
                                  STEP = STEP)


print ('Single window of past history : {}'.format(x_train[0].shape))

# %% TRAIN VAL DATA
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# train_data = train_data.cache().shuffle(BUFFER_SIZE).repeat()
train_data = train_data.cache().batch(BATCH_SIZE).repeat()

val_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data = val_data.batch(BATCH_SIZE).repeat()

# %%

model = NAX_model(INPUT_SHAPE=x_train.shape[-2:],
            REG_PARAM = REG_PARAM,
            ACT_FUN = ACT_FUN,
            LEARN_RATE = LEARN_RATE,
            HIDDEN_NEURONS=HIDDEN_NEURONS ,
            OUTPUT_NEURONS= OUTPUT_NEURONS,
            LOSS_FUNCTION =  LOSS_FUNCTION)


# %%

for x, y in val_data.take(1):
  print(model.predict(x).shape)

# %%
EARLYSTOP = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=STOPPATIENCE)
history=model.fit(
    x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[EARLYSTOP],
    validation_data=(x_val,y_val), validation_batch_size=BATCH_SIZE,
)

plot_train_history(history,"Loss of model")
# %%
y_pred =model.predict(x_val)
N_val = len(y_pred)
the_length = len(df['std_demand'])
START = TRAIN_SPLIT+past_history+future_target



std_demand_true = pd.Series(df['std_demand'][START:START+N_val])
std_demand_NAX = (std_demand_true- y_val)+y_pred[:,0]
std_demand_GLM = (std_demand_true- y_val)


# %%
demand_log_train= np.exp( pd.Series(df['log_demand'][START_SPLIT:TRAIN_SPLIT]) )

demand_true=inverse_std(std_demand_true,demand_log_train) 
demand_NAX=inverse_std(std_demand_NAX,demand_log_train) 
demand_GLM=inverse_std(std_demand_GLM,demand_log_train) 

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
