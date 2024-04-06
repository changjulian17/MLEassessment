from base import BaseMLP
import numpy as np


class NeuralNetworkClassifier(BaseMLP):
    """
    Classifier using a neural network.
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
        self.create_layers()

    def create_layers(self):
        self.W[1] = np.random.rand(10, 2) - 0.5
        self.b[1] = np.random.rand(10, 1) - 0.5
        self.W[2] = np.random.rand(10, 2) - 0.5
        self.b[2] = np.random.rand(10, 1) - 0.5
        self.W[3] = np.random.rand(10, 10) - 0.5
        self.b[3] = np.random.rand(10, 1) - 0.5
        self.W[4] = np.random.rand(2, 10) - 0.5
        self.b[4] = np.random.rand(2, 1) - 0.5
        return self.W, self.b

    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def softmax(self, Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def forward_prop(self, X):
        self.Z[1] = self.W[1].dot(X.T) + self.b[1]
        self.A[1] = self.ReLU(self.Z[1])
        self.Z[2] = self.W[2].dot(X.T) + self.b[2]
        self.A[2] = self.ReLU(self.Z[2])
        self.Z[3] = self.W[3].dot(self.A[2]) + self.b[3]
        self.A[3] = self.ReLU(self.Z[3])
        Z4 = self.W[4].dot(self.A[3]) + self.b[4]
        A4 = self.softmax(Z4)
        return self.A, self.Z

    def ReLU_deriv(self, Z):
        return Z > 0

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, 2))
        one_hot_Y[np.arange(Y.size).astype(int), Y.astype(int)] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(self, X, Y):
        one_hot_Y = self.one_hot(Y)
        self.dZ[4] = self.A[4] - one_hot_Y
        self.dW[4] = 1 / X.shape[0] * self.dZ[4].dot(self.A[3].T)
        self.db[4] = 1 / X.shape[0] * np.sum(self.dZ[4])
        self.dZ[3] = self.W[4].T.dot(self.dZ[4]) * self.ReLU_deriv(self.Z[3])
        self.dW[3] = 1 / X.shape[0] * self.dZ[3].dot(self.A[2].T)
        self.db[3] = 1 / X.shape[0] * np.sum(self.dZ[3])
        # Similar computation for dZ2, dW2, db2, dZ1, dW[1], db1
        return self.db, self.dW

    def update_params(self, alpha):
        for i, weight in enumerate(self.W[1:]):  # update weights
            weight -= alpha * self.dW[i]

        for i, bias in enumerate(self.b[1:]):  # update biases
            bias -= alpha * self.db[i]

    def train(self, X, Y, alpha, iterations):
        for i in range(iterations):
            self.Z, selfA = self.forward_prop(X)
            self.db, self.dW = self.backward_prop(X, Y)
            self.update_params(alpha)
            if i % 1000 == 0:
                predictions = np.argmax(self.A[4], axis=0)
                accuracy = np.sum(predictions == Y) / Y.size
                print(f"Iteration: {i}, Accuracy: {accuracy}")

    def make_predictions(self, X):
        self.A, self.Z = self.forward_prop(X)
        predictions = np.argmax(self.A[4], axis=0)
        pass

    def fit(self, x, y) -> None:
        """
        Fit the model to data matrix X and target(s) Y.

        Parameters
        ----------
        x : list or sparse matrix of shape (n_samples, n_features)
            The input data
        y : list of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels for classification).
        """

    def predict(self, x):
        """
        Predict using the neural network classifier.

        Parameters
        ----------
        x : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y : list, shape (n_samples) or  (n_samples, n_classes)
            The predicted classes.
        """
        # Your implementation for predicting with the neural network classifier goes here
        pass
