from abc import ABC, abstractmethod
import numpy as np
from base import BaseMLP


def init_params():
    W1 = np.random.rand(10, 2) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    W3 = np.random.rand(10, 10) - 0.5
    b3 = np.random.rand(10, 1) - 0.5
    W4 = np.random.rand(2, 10) - 0.5
    b4 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2, W3, b3, W4, b4


def ReLU(Z):  # returns Z for Z > 0
    return np.maximum(Z, 0)


def softmax(Z):  # returns a positive no. less than 1
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 2))
    one_hot_Y[np.arange(Y.size).astype(int), Y.astype(int)] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X):
    Z1 = W1.dot(X.T) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = ReLU(Z3)
    Z4 = W4.dot(A3) + b4
    A4 = softmax(Z4)
    return Z1, A1, Z2, A2, Z3, A3, Z4, A4


def backward_prop(Z1, A1, Z2, A2, Z3, A3, Z4, A4, W1, W2, W3, W4, X, Y):
    m = Y.shape[0]
    one_hot_Y = one_hot(Y)  # loss
    dZ4 = A4 - one_hot_Y
    dW4 = 1 / m * dZ4.dot(A3.T)
    db4 = 1 / m * np.sum(dZ4)
    dZ3 = W4.T.dot(dZ4) * ReLU_deriv(Z3)
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3)
    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3, dW4, db4


def update_params(W1, b1, W2, b2, W3, b3, W4, b4, dW1, db1, dW2, db2, dW3, db3, dW4, db4, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3
    W4 = W4 - learning_rate * dW4
    b4 = b4 - learning_rate * db4
    return W1, b1, W2, b2, W3, b3, W4, b4


def get_predictions(A4):
    return np.argmax(A4, 0)


def get_accuracy(predictions, Y):
    print(predictions)
    return np.sum(predictions == Y) / Y.size


class NeuralNet(BaseMLP):
    """
    Base class for MLP classification.
    """

    def __init__(self, hidden_layer_sizes, batch_size, learning_rate, max_iter, random_state, momentum):
        super().__init__(hidden_layer_sizes, batch_size, learning_rate, max_iter, random_state, momentum)

    def fit(self, X, Y) -> None:
        """
        Fit the model to data matrix X and target(s) Y.

        Parameters
        ----------
        X : list or sparse matrix of shape (n_samples, n_features)
            The input data
        Y : list of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels for classification).
        """
        W1, b1, W2, b2, W3, b3, W4, b4 = init_params()

        for i in range(self.max_iter):
            Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
            dW1, db1, dW2, db2, dW3, db3, dW4, db4 = backward_prop(Z1, A1, Z2, A2, Z3, A3, Z4, A4, W1, W2, W3, W4, X, Y)
            W1, b1, W2, b2, W3, b3, W4, b4 = update_params(W1, b1, W2, b2, W3, b3, W4, b4, dW1, db1, dW2, db2, dW3, db3,
                                                           dW4, db4, self.learning_rate)
            if i % 10000 == 0:
                print("Iteration: ", i)
                print("Accuracy:  ", get_accuracy(get_predictions(A4), Y))

        return W1, b1, W2, b2, W3, b3, W4, b4


    def predict(self, W1, b1, W2, b2, W3, b3, W4, b4, X):
        """
        Predict using the multi-layer perceptron classifier.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y : list, shape (n_samples) or  (n_samples, n_classes)
            The predicted classes.
        """

        Z1, A1, Z2, A2, Z3, A3, Z4, A4 = forward_prop(W1, b1, W2, b2, W3, b3, W4, b4, X)
        predictions = get_predictions(A4)
        return A4
