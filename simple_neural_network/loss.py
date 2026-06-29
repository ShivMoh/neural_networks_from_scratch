import numpy as np
from nn.activation import Softmax
class Cross_Entropy_Loss():
    def compute(self, softmax_outputs, class_targets):
        loss = 0
        if len(class_targets.shape) == 1:
            loss = softmax_outputs[range(len(softmax_outputs)), class_targets]
        if len(class_targets.shape) == 2:
            loss = np.sum(softmax_outputs * class_targets, axis=1)

        # loss = -np.log(loss)

        loss = -np.log(np.clip(loss, 1e-7, 1 - 1e-7))

        mean_loss = np.mean(loss)

        return mean_loss

class Activation_Softmax_Loss_CategoricalCrossEntropy():

    def __init__(self):
        self.activation = Softmax()
        self.loss = Cross_Entropy_Loss()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output

        return self.loss.compute(self.output, y_true)

    def backward(self, dvalues, y_true):
        samples = len(dvalues)

        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs = self.dinputs / samples
        
        return self.dinputs