import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
from MLE_loss import loss_strike
from tensorflow.keras.callbacks import EarlyStopping
from NAX_f import prep_data, aggregate_data, NAX_model, demands
from standard_and_error_functions import rmse

# %% Define the parameters
my_loss = loss_strike(.005)     # custom loss function

MAX_EPOCHS = 500    # number of epochs for the training
STOPPATIENCE = 50   # number of iterations before Keras callback stops the function

# Possible values of hyper-parameters
LIST_HIDDEN_NEURONS = [3, 4, 5, 6]      # number of neurons (hidden layer)
LIST_ACT_FUN = ['softmax', 'sigmoid']   # activation function
LIST_LEARN_RATE = [0.1, 0.01, 0.003, 0.001]     # initial learning rate (for Keras ADAM)
LIST_REG_PARAM = [0.001, 0.0001, 0]     # regularization parameter
LIST_BATCH_SIZE = [50, 5000]     # batch size, 5000 for no batch

# How to split the data
START_SPLIT = 0     # first row of the training set
TRAIN_SPLIT = 1095  # last row of the training set
END_SPLIT = 1095+365    # last row of the validation set

past_history = 1    # number of past values to be used
future_target = 0   # number of future values to predict

VERBOSE = 1
VERBOSE_EARLY = 1

# %%
def find_hyperparam(df_NAX, M, m,

                    LOSS_FUNCTION = my_loss,
                    MAX_EPOCHS = MAX_EPOCHS,
                    STOPPATIENCE = STOPPATIENCE,

                    LIST_HIDDEN_NEURONS = LIST_HIDDEN_NEURONS,
                    LIST_ACT_FUN = LIST_ACT_FUN,
                    LIST_LEARN_RATE = LIST_LEARN_RATE,
                    LIST_REG_PARAM = LIST_REG_PARAM,
                    LIST_BATCH_SIZE = LIST_REG_PARAM,
                    
                    START_SPLIT = START_SPLIT,
                    TRAIN_SPLIT = TRAIN_SPLIT,
                    END_SPLIT = END_SPLIT,

                    past_history = past_history,
                    future_target = future_target,
                    
                    VERBOSE = VERBOSE,
                    VERBOSE_EARLY = VERBOSE_EARLY):
    
    # This function selects the optimal hyper-parameters (corresponding to the minimum RMSE)
    #
    # INPUTS:
    # df_NAX
    # previously defined parameters
    # LIST_HIDDEN_NEURONS:  number of neurons (hidden layer)
    # LIST_ACT_FUN:         activation function
    # LIST_LEARN_RATE:      initial learning rate (for Keras ADAM)
    # LIST_REG_PARAM:       regularization parameter
    # LIST_BATCH_SIZE:      batch size
    # M:        maximum of the log_demand observed over the years: 2008 - 2016
    # m:        minimum of the log_demand observed over the years: 2008 - 2016
    #
    # OUTPUTS:
    # hyper_parameters: selected hyper-parameters
    # min_RMSE:         minimum RMSE, corresponding to the selected hyper-parameters
    # RMSE:             RMSE corresponding to all possible combinations of hyper-parameters
    # grid_history:     last element of the validation loss of each possible combinations of hyper-parameters

    # get the needed features and the corresponding labels
    features, labels = prep_data(df_NAX,
                       START_SPLIT = START_SPLIT,
                       TRAIN_SPLIT = TRAIN_SPLIT,
                       END_SPLIT = END_SPLIT)
    
    # aggregate data from features and labels in couples in the structure (x(t-1), x(t)), (y(t))
    x_train, y_train, x_val, y_val = aggregate_data(features,labels,
                                     START_SPLIT = START_SPLIT,
                                     TRAIN_SPLIT = TRAIN_SPLIT,
                                     END_SPLIT = END_SPLIT,
                                     past_history = past_history,
                                     future_target = future_target)

    START = TRAIN_SPLIT + past_history + future_target - 1
    # early stopping criterium
    EARLYSTOP = EarlyStopping(monitor='val_loss', mode='min', verbose=VERBOSE_EARLY, patience=STOPPATIENCE)

    L1,L2,L3,L4,L5 = len(LIST_HIDDEN_NEURONS), len(LIST_ACT_FUN), len(LIST_LEARN_RATE), len(LIST_REG_PARAM), len(LIST_BATCH_SIZE)

    N_SCENARIOS = L1*L2*L3*L4*L5

    RMSE = np.zeros((L1,L2,L3,L4,L5))
    grid_history = np.zeros((L1,L2,L3,L4,L5))
    c = 0
    for n1 in range(L1):
        for n2 in range(L2):
            for n3 in range(L3):
                for n4 in range(L4):
                    for n5 in range(L5):
                        # for every possible combination of hyper-parameters
                        # build a simpleRNN model
                        model = NAX_model(INPUT_SHAPE = x_train.shape[-2:],
                                          REG_PARAM = LIST_REG_PARAM[n4],
                                          ACT_FUN = LIST_ACT_FUN[n2],
                                          LEARN_RATE = LIST_LEARN_RATE[n3],
                                          HIDDEN_NEURONS = LIST_HIDDEN_NEURONS[n1],
                                          LOSS_FUNCTION = LOSS_FUNCTION)
                        # train the model
                        history = model.fit(x = x_train, y = y_train, 
                                            batch_size = LIST_BATCH_SIZE[n5], epochs = MAX_EPOCHS, 
                                            verbose = 0, callbacks = [EARLYSTOP], 
                                            validation_data = (x_val, y_val), validation_batch_size = LIST_BATCH_SIZE[n5])
                        # predict output corresponding to input x_val
                        y_pred = model.predict(x_val)
                        
                        # consider realised demand and compute demand in original units from the NAX prediction
                        demand_true, demand_NAX, _ = demands(y_pred, y_val, df_NAX, START, M, m)

                        # compute RMSE
                        RMSE[n1][n2][n3][n4][n5] = rmse(demand_NAX, demand_true)

                        # print progress of the loop
                        if VERBOSE==1:
                            RMSE[n1][n2][n3][n4][n5] = rmse(demand_NAX,demand_true)
                            hyper_parameters = [LIST_HIDDEN_NEURONS[n1],
                                                LIST_ACT_FUN[n2], 
                                                LIST_LEARN_RATE[n3], 
                                                LIST_REG_PARAM[n4],
                                                LIST_BATCH_SIZE[n5]]

                            c +=1
                            print(c, " / ", N_SCENARIOS)
                            print(rmse(demand_NAX,demand_true), ' --', hyper_parameters)
                            # print(round(c/N_SCENARIOS,4)*100,'%)

                        grid_history[n1][n2][n3][n4][n5] = history.history['val_loss'][-1]                   

    # select the optimal hyper-parameters (corresponding to the minimum RMSE)
    argmin = np.unravel_index(np.argmin(RMSE,axis=None),RMSE.shape)
    hyper_parameters = [LIST_HIDDEN_NEURONS[argmin[0]],
                        LIST_ACT_FUN[argmin[1]], 
                        LIST_LEARN_RATE[argmin[2]], 
                        LIST_REG_PARAM[argmin[3]],
                        LIST_BATCH_SIZE[argmin[4]]]
    
    # select the minimum RMSE
    min_RMSE = np.min(RMSE,axis=None)

    return hyper_parameters, min_RMSE, RMSE, grid_history