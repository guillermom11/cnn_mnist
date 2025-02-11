import numpy as np
 
# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2))

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size #same as -2(y_true-y_pred)/y_true.size. A bit different from part b) since the Loss function is just (y - y_predicted)^2