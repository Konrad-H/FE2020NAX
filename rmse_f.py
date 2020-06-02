def rmse(predictions, targets):
    from numpy import sqrt
    return sqrt(((predictions - targets) ** 2).mean())