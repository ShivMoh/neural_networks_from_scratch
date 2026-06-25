import numpy as np
def Cross_Entropy_Loss():
    def __init__():
        pass

    def compute_loss(self, softmax_outputs, class_targets):
        loss = 0
        if class_targets.shape == 1:
            loss = softmax_outputs[[range(len(softmax_outputs))], class_targets]
        if class_targets.shape == 2:
            loss = np.sum(softmax_outputs * class_targets, axis=1)

        loss = -np.log(loss)

        mean_loss = np.mean(loss)

        return mean_loss

