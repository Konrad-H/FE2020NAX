import tensor flow as tf
def RNN_model(REG_PARAM,OUTPUT_NEURONS,HIDDEN_NEURONS,LOSS_FUNCTION,LEARN_RATE)
   
    opt = tf.keras.optimizers.Adam(LEARN_RATE)
    act_reg = tf.keras.regularizers.l1 (REG_PARAM)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.SimpleRNN(HIDDEN_NEURONS,
                                            input_shape=x_train.shape[-2:],
                                            activation=ACT_FUN,
                                            activity_regularizer= act_reg ,
                                            #  kernel_initializer=tf.keras.initializers.RandomNormal(stddev=1),
                                            # bias_initializer=tf.keras.initializers.Ones()
                                            ))                                            
    model.add(tf.keras.layers.Dense(OUTPUT_NEURONS))

    # opt = tf.keras.optimizers.RMSprop()
    model.compile(optimizer=opt, loss=LOSS_FUNCTION)
    return model