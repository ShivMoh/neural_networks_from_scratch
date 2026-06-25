# the class for Schotastic Gradient Descent
class SGD():
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate

    def update_params(self, layer, lr):
        layer.weights = layer.weights - (lr * layer.dweights)
        layer.biases = layer.biases - (lr * layer.dbiases)

        return layer.weights, layer.biases
