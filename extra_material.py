#perceptron with central patterns
from scipy.stats import norm
import numpy as np

#Train and test perceptron
def train_perceptron(A_R, labels, epochs, eta):
    N_samples, N_r = A_R.shape             #A_R is the matrix of ONLY central patterns
    W = np.random.normal(0, 1 / np.sqrt(N_r), size=N_r) #initialize the weight vector W
    f = np.mean(A_R)   #fraction of active neurons

    for epoch in range(epochs):
        errors = 0
        for mu in np.random.permutation(N_samples):
            x_mu = A_R[mu]        # input pattern, central pattern
            y_mu = labels[mu]     # target label
            y_pred = np.sign(np.dot(x_mu - f, W)) #prediction
            if y_pred == 0:
                y_pred = 1
            if y_mu * np.dot((x_mu-f), W) <= 0: #if prediction is wrong update (HERE WE SHOULD USE ONLY CENTRAL PATTERN)
                W += eta * y_mu * x_mu
                errors += 1
        if errors == 0:
            break
    return W

def test_perceptron(A_R, labels, W):
    preds = np.sign((A_R - f) @ W)
    preds[preds == 0] = 1
    error = 1 - np.mean(preds == labels)
    return error