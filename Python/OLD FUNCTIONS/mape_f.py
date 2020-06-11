def mape(y, y_pred):
    from numpy import abs, mean
    return mean(abs((y-y_pred)/y))