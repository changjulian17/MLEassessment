

class NeuralNetCV():
    def __init__(self, k_folds=5, **kwargs):
        super().__init__(**kwargs)
        self.k_folds = k_folds



# Example usage:
model = NeuralNetCV(hidden_layer_sizes=10, batch_size=32, learning_rate=0.01, max_iter=1000, random_state=42, momentum=0.9)
model.cross_validate(X, Y)
