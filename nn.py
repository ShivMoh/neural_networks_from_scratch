import numpy as np

class Linear():
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)

class Flatten():
    def forward(self, x):
        self.original_shape = x.shape
        if x.ndim == 1:
            return x.reshape(1, -1)  # single sample: (64,) → (1, 64)
        return x.reshape(x.shape[0], -1)  # batch: (N, H, W) → (N, H*W)

    def backward(self, grad):
        return grad.reshape(self.original_shape)
