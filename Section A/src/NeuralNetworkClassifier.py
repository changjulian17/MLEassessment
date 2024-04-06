import numpy as np
from base import BaseMLP


def init_params():
    W = [*range(10)]
    b = [*range(10)]
    W[1] = np.random.rand(10, 2) - 0.5
    b[1] = np.random.rand(10, 1) - 0.5
    W[2] = np.random.rand(10, 10) - 0.5
    b[2] = np.random.rand(10, 1) - 0.5
    W[3] = np.random.rand(10, 10) - 0.5
    b[3] = np.random.rand(10, 1) - 0.5
    W[4] = np.random.rand(2, 10) - 0.5
    b[4] = np.random.rand(2, 1) - 0.5

    return W, b


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


def forward_prop(b: list, W: list, X):
    layers = 5  # todo change when adding layers

    A = [*range(len(W))]
    Z = [*range(len(W))]

    # input layer
    Z[1] = W[1].dot(X.T) + b[1]
    A[1] = ReLU(Z[1])

    # hidden layers
    for layer in range(2, layers-1):
        Z[layer] = W[layer].dot(A[layer - 1]) + b[layer]
        A[layer] = ReLU(Z[layer])

    # output layer
    Z[4] = W[4].dot(A[3]) + b[4]
    A[4] = softmax(Z[4])
    return A, Z


def backward_prop(A: list, W: list, Z: list, X, Y):
    layers = 5  # todo change when added layers
    db = [*range(1, layers + 1)]
    dW = [*range(1, layers + 1)]
    dZ = [*range(1, layers + 1)]
    m = Y.shape[0]
    one_hot_Y = one_hot(Y)

    # output layer
    dZ[4] = A[4] - one_hot_Y  # loss
    dW[4] = 1 / m * dZ[4].dot(A[3].T)
    db[4] = 1 / m * np.sum(dZ[4])

    # hidden layers
    for layer in [3, 2]:
        dZ[layer] = np.dot(W[layer+1].T, dZ[layer+1]) * ReLU_deriv(Z[layer])
        dW[layer] = 1 / m * dZ[layer].dot(A[layer-1].T)
        db[layer] = 1 / m * np.sum(dZ[layer])

    # output layer
    dZ[1] = W[2].T.dot(dZ[2]) * ReLU_deriv(Z[1])
    dW[1] = 1 / m * dZ[1].dot(X)
    db[1] = 1 / m * np.sum(dZ[1])

    return db, dW


def update_params(b: list, W: list, db: list, dW: list, learning_rate):
    W[1] = W[1] - learning_rate * dW[1]
    b[1] = b[1] - learning_rate * db[1]
    W[2] = W[2] - learning_rate * dW[2]
    b[2] = b[2] - learning_rate * db[2]
    W[3] = W[3] - learning_rate * dW[3]
    b[3] = b[3] - learning_rate * db[3]
    W[4] = W[4] - learning_rate * dW[4]
    b[4] = b[4] - learning_rate * db[4]
    return W, b


def get_predictions(output_layer):
    return np.argmax(output_layer, 0)


def get_accuracy(predictions, Y):
    print(predictions)
    return np.sum(predictions == Y) / Y.size


class NeuralNet(BaseMLP):
    """
    Base class for MLP classification.
    """

    def __init__(self, hidden_layer_sizes, batch_size, learning_rate, max_iter, random_state, momentum):
        super().__init__(hidden_layer_sizes, batch_size, learning_rate, max_iter, random_state, momentum)
        self.b = [*range(10)]
        self.Z = [*range(10)]
        self.W = [*range(10)]
        self.A = [*range(10)]
        self.dW = [*range(10)]
        self.db = [*range(10)]
        self.dZ = [*range(10)]

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
        self.W, self.b = init_params()

        for i in range(self.max_iter):
            self.A, self.Z = forward_prop(self.b, self.W, X)
            self.db, self.dW = backward_prop(self.A, self.W, self.Z, X, Y)
            self.W, self.b = update_params(self.b, self.W, self.db, self.dW, self.learning_rate)
            if i % 100 == 0:
                print("Iteration: ", i)
                print("Accuracy:  ", get_accuracy(get_predictions(self.A[4]), Y))    #TODO CHANGE TO LAST LAYER

        return None

    def predict(self, x):
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

        self.A, self.Z = forward_prop(self.b, self.W, x)
        predictions = get_predictions(self.A[4])  # todo change to output layer

        return predictions
