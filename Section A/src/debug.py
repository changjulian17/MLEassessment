import sys
sys.path.insert(0, 'src')  # Add the 'src' directory to the Python path
from NeuralNetworkClassifier import NeuralNet, get_accuracy  # Import the module
from get_data import get_data


Y_dev, X_dev, Y_train, X_train = get_data(1000, 0.1, split=0.8)

print(Y_dev.shape, X_dev.shape, Y_train.shape, X_train.shape)


mdl = NeuralNet(
    hidden_layer_sizes=10,
    batch_size=1,
    learning_rate=.08,
    max_iter=20_000,
    random_state=42,
    momentum=1
)


mdl.fit(X_train, Y_train)

predictions = mdl.predict(X_dev)

print(get_accuracy(predictions, Y_dev))
print("done")


