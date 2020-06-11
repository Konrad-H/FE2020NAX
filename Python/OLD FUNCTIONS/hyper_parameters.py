# %% import packages
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf

import numpy as np
import pandas as pd 


from NAX_functions import custom_loss, inverse_std
from tensorflow.keras.callbacks import EarlyStopping

from NAX_f import prep_data, aggregate_data, NAX_model, demands, rmse


# %% load dataset
df_NAX = pd.read_csv("train_data.csv",index_col=0) 
df_NAX.head()   # length 1095

START_SPLIT = 0
TRAIN_SPLIT = 1095
VAL_SPLIT = 1095+365
past_history = 2
future_target = -1
STEP = 1

LIST_ACT_FUN = ['softmax', 'sigmoid'] #['softmax', 'sigmoid']
LIST_HIDDEN_NEURONS = [3, 4, 5, 6] #[3, 4, 5, 6]
LIST_LEARN_RATE = [0.1, 0.01, 0.003, 0.001] #[0.1, 0.01, 0.003, 0.001]
LIST_BATCH_SIZE = [50,5000] # manca no batch, None non funziona
LIST_REG_PARAM = [0.001, 0.0001, 0]


features,labels= prep_data(df_NAX,
                    START_SPLIT = START_SPLIT,
                    TRAIN_SPLIT = TRAIN_SPLIT,
                    VAL_SPLIT = VAL_SPLIT)
x_train, y_train,x_val, y_val = aggregate_data(features,labels,
                                  START_SPLIT = START_SPLIT,
                                  TRAIN_SPLIT = TRAIN_SPLIT,
                                  VAL_SPLIT = VAL_SPLIT,
                                  past_history = past_history,
                                  future_target = future_target,
                                  STEP = STEP)




# %%
MAX_EPOCHS = 500 #
STOPPATIENCE = 10
#EVALUATION_INTERVAL = 500

LIST_LOSS_FUNCTION = [custom_loss, 'mse']
LOSS_FUNCTION = LIST_LOSS_FUNCTION[0]

START = TRAIN_SPLIT+past_history+future_target
EARLYSTOP = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=STOPPATIENCE)



# %% 

#[0.001, 0.0001, 0]
L1,L2,L3,L4,L5=len(LIST_HIDDEN_NEURONS), len(LIST_ACT_FUN), len(LIST_LEARN_RATE), len(LIST_REG_PARAM), len(LIST_BATCH_SIZE)

N_SCENARIOS = L1*L2*L3*L4*L5

# %% search optimal hyper parameters
RMSE = np.zeros((L1,L2,L3,L4,L5, ))
c = 0
for n1 in range(L1):
    for n2 in range(L2):
        for n3 in range(L3):
            for n4 in range(L4):
                for n5 in range(L5):
                    model = NAX_model(INPUT_SHAPE=x_train.shape[-2:],
                                        REG_PARAM = LIST_REG_PARAM[n4],
                                        ACT_FUN = LIST_ACT_FUN[n2],
                                        LEARN_RATE = LIST_LEARN_RATE[n3],
                                        HIDDEN_NEURONS=LIST_HIDDEN_NEURONS[n1] ,
                                        LOSS_FUNCTION =  LOSS_FUNCTION)
                    model.fit(x=x_train, y=y_train, 
                            batch_size=LIST_BATCH_SIZE[n5], epochs=MAX_EPOCHS, 
                            verbose=0, callbacks=[EARLYSTOP], 
                            validation_data=(x_val,y_val), validation_batch_size=LIST_BATCH_SIZE[n5],
                            ) #shuffle=True,
                    y_pred =model.predict(x_val)
                    
                    demand_true, demand_NAX, _ = demands(y_pred,y_val, df_NAX,START)
                    print(rmse(demand_NAX,demand_true))
                    RMSE[n1][n2][n3][n4][n5]=rmse(demand_NAX,demand_true)
                    c +=1
                    print(round(c/N_SCENARIOS,4)*100,'%')

# %%
                    

argmin = np.unravel_index(np.argmin(RMSE,axis=None),RMSE.shape)
hyper_parameters = [ LIST_HIDDEN_NEURONS[argmin[0]],
                    LIST_ACT_FUN[argmin[1]], 
                    LIST_LEARN_RATE[argmin[2]], 
                    LIST_REG_PARAM[argmin[3]],
                    LIST_BATCH_SIZE[argmin[4]] ]

print('\n Hyper parameters')
print(hyper_parameters)
print('RMSE_NAX',np.min(RMSE,axis=None))

# it should be:
# HIDDEN_NEURONS = 3
# ACT_FUN = 'softmax'
# LEARN_RATE = 0.003
# REG_PARAM = 10^-4
# BATCH_SIZE = 50

# seed 14
# clip 0.008 FAILURE
# clip 0.007 FAILURE ([6, 'softmax', 0.003, 0.0001, 50])
# clip 0.0065 FAILURE ([5, 'softmax', 0.003, 0.0001, 50])
# clip 0.006 FAILURE ([5, 'softmax', 0.003, 0.001, 50])
# clip 0.0016 FAILURE ([6, 'sigmoid', 0.01, 0, 5000])
# clip 0.002 FAILURE ([5, 'sigmoid', 0.01, 0.0001, 50])
# clip 0.003 FAILURE ([5, 'sigmoid', 0.003, 0.001, 50])
# clip 0.005 FAILURE ([5, 'sigmoid', 0.003, 0, 50])

# seed 50
# clip 0.004 FAILURE ([3, 'sigmoid', 0.003, 0.001, 50])
# clip 0.005 FAILURE ([4, 'softmax', 0.01, 0.0001, 5000])

# %%
