from sklearn.datasets import make_circles
import numpy as np


def get_data(n_samples=1000, noise=0.1, split=0.8):
    X, Y = make_circles(n_samples=n_samples, noise=noise)

    # Combine X and Y into a single array along the columns axis
    data_combined = np.column_stack((Y, X))

    # Shuffle the combined data along the rows axis
    np.random.shuffle(data_combined)

    m, n = data_combined.shape
    divider = int(m * (1-split)) + 1

    # make dev data
    data_dev = data_combined[0:divider]
    Y_dev = data_dev[:, 0]
    X_dev = data_dev[:, 1:n]

    # make train data
    data_train = data_combined[divider:]
    Y_train = data_train[:, 0]
    X_train = data_train[:, 1:n]
    _, m_train = X_train.shape

    return Y_dev, X_dev, Y_train, X_train
