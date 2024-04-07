import numpy as np
from base import BaseMLP
from sklearn.model_selection import KFold


def init_params():
    W = [*range(10)]
    b = [*range(10)]  # W0 is input

    W[1] = np.random.rand(10, 2) - 0.5  # Hidden 1
    b[1] = np.random.rand(10, 1) - 0.5
    W[2] = np.random.rand(10, 10) - 0.5  # Dropout 1
    b[2] = np.random.rand(10, 1) - 0.5
    W[3] = np.random.rand(10, 10) - 0.5  # hidden 2
    b[3] = np.random.rand(10, 1) - 0.5
    W[4] = np.random.rand(10, 10) - 0.5  # Dropout 2
    b[4] = np.random.rand(10, 1) - 0.5
    W[5] = np.random.rand(10, 10) - 0.5  # hidden 3
    b[5] = np.random.rand(10, 1) - 0.5
    W[6] = np.random.rand(10, 10) - 0.5  # Dropout 3
    b[6] = np.random.rand(10, 1) - 0.5
    W[7] = np.random.rand(10, 10) - 0.5  # hidden 4
    b[7] = np.random.rand(10, 1) - 0.5
    W[8] = np.random.rand(10, 10) - 0.5  # Dropout 4
    b[8] = np.random.rand(10, 1) - 0.5

    W[9] = np.random.rand(2, 10) - 0.5  # Output
    b[9] = np.random.rand(2, 1) - 0.5

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
    layers = 10

    A = [*range(len(W))]
    Z = [*range(len(W))]

    # input layer
    Z[1] = W[1].dot(X.T) + b[1]
    A[1] = ReLU(Z[1])

    # hidden layers
    for layer in [*range(2, layers - 1)]:
        Z[layer] = W[layer].dot(A[layer - 1]) + b[layer]
        A[layer] = ReLU(Z[layer])

    # output layer
    Z[9] = W[9].dot(A[8]) + b[9]
    A[9] = softmax(Z[9])
    return A, Z


def backward_prop(A: list, D: list, W: list, Z: list,  X, Y):
    # Assume a drop out of 0.0.3
    p = 0.3
    layers = len(W)
    db = [*range(1, layers+1)]
    dW = [*range(1, layers+1)]
    dZ = [*range(1, layers+1)]
    m = Y.shape[0]
    one_hot_Y = one_hot(Y)

    # output layer
    dZ[9] = A[9] - one_hot_Y  # loss
    dW[9] = 1 / m * dZ[9].dot(A[8].T)
    db[9] = 1 / m * np.sum(dZ[9])

    # hidden and dropout layers
    for layer in [*range(2, layers - 1)][::-1]:    # start from back to front
        if layer in {2, 4, 6, 8}:
            # logic for dropout
            D[layer] = np.random.rand(A[layer].shape[0], A[layer].shape[1])
            D[layer] = D[layer] < p
            dA = np.dot(W[layer + 1].T, dZ[layer + 1]) * D[layer]  # Step 1: Apply mask D2
            dZ[layer] = dA * ReLU_deriv(Z[layer]) / (1 - p)  # Step 2: Scale the value of active neurons
            dW[layer] = 1 / m * dZ[layer].dot(A[layer - 1].T)
            db[layer] = 1 / m * np.sum(dZ[layer])
        dZ[layer] = np.dot(W[layer + 1].T, dZ[layer + 1]) * ReLU_deriv(Z[layer])
        dW[layer] = 1 / m * dZ[layer].dot(A[layer - 1].T)
        db[layer] = 1 / m * np.sum(dZ[layer])

    # output layer
    dZ[1] = W[2].T.dot(dZ[2]) * ReLU_deriv(Z[1])
    dW[1] = 1 / m * dZ[1].dot(X)
    db[1] = 1 / m * np.sum(dZ[1])

    return db, dW


def update_params(b: list, W: list, db: list, dW: list, learning_rate):
    layers = len(W)
    for layer in [*range(1,10)]:
        W[layer] = W[layer] - learning_rate * dW[layer]
        b[layer] = b[layer] - learning_rate * db[layer]
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
        self.D = [*range(10)]

    def fit(self, X, Y) -> None:
        """
        Fit the model to data matrix X and target(s) Y.

        Parameters
        ----------
        X : list or sparse matrix of shape (n_samples, n_features)
            The input data
        Y : list of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels for classification).
        batch_size : int, optional (default=32)
            The size of each mini-batch.
        random_state : int or None, optional (default=None)
            The random seed for reproducible sampling of mini-batches.
        """
        # Set random seed if provided
        np.random.seed(self.random_state)
        self.W, self.b = init_params()

        # Determine the number of batches
        n_samples = X.shape[0]
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size

        for i in range(self.max_iter):
            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min((batch + 1) * self.batch_size, n_samples)
                X_batch = X[start_idx:end_idx]
                Y_batch = Y[start_idx:end_idx]

                self.A, self.Z = forward_prop(self.b, self.W, X_batch)
                self.db, self.dW = backward_prop(self.A, self.D, self.W, self.Z, X_batch, Y_batch)
                self.W, self.b = update_params(self.b, self.W, self.db, self.dW, self.learning_rate)
                if i % 100 == 0:
                    print("Iteration: ", i)
                    print("Accuracy:  ", get_accuracy(get_predictions(self.A[9]), Y))  # TODO CHANGE TO LAST LAYER

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
        predictions = get_predictions(self.A[9])  # todo change to output layer

        return predictions

    def cross_validate(self, X, Y, k_folds):
        kf = KFold(n_splits=k_folds)
        accuracies = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            self.fit(X_train, Y_train)
            predictions = self.predict(X_test)
            accuracy = get_accuracy(predictions, Y_test)
            accuracies.append(accuracy)

        avg_accuracy = sum(accuracies) / len(accuracies)
        print(f"Average Accuracy: {avg_accuracy}")
