from nn.nn import Linear, Flatten
from nn.activation import ReLU, Softmax
from nn.loss import Cross_Entropy_Loss, Activation_Softmax_Loss_CategoricalCrossEntropy
from nn.optimizer import SGD
from nn.data import convert_ds_to_numpy
from nn.plot import plot_loss, plot_predictions
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


if __name__ == "__main__":
    print("I am testing to see if this works")

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch size 1 for prediction

    labels, images = convert_ds_to_numpy(train_loader)

    flatten = Flatten()
    linear = Linear(784, 10)
    loss = Cross_Entropy_Loss()
    softmax = Softmax()
    optimizer = SGD(1)
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()
    losses = []

    for label, image in zip(labels, images):
        # print("Image", image.shape)
        # print("class label", label)
        
        
        x = flatten.forward(image)

        x = linear.forward(x)
        # softmax_output = softmax.forward(x)
        # loss = loss.compute(softmax_output, label)
        
        activation_loss = softmax_loss.forward(x, label)
        losses.append(activation_loss)
        
        dvalues = softmax_loss.backward(softmax_loss.output, label)
        dvalues = linear.backward(dvalues)

        # print("before updates", linear.weights)
        # print("before updates", linear.biases)

        optimizer.update_params(linear, 1)
        
        # print("after updates", linear.weights)
        # print("after updates", linear.biases)
        
        # dvalues = loss.backward(x, label)
    
    # plot_loss(losses)    
    
    test_labels, test_images = convert_ds_to_numpy(test_loader)

    correct = 0
    total = 0
    sample_images, sample_true, sample_pred = [], [], []

    for label, image in zip(test_labels, test_images):
        x = flatten.forward(image)
        x = linear.forward(x)
        x = softmax.forward(x)

        predicted = np.argmax(x, axis=1)

        if predicted == label:
            correct += 1
        total += 1

        if len(sample_images) < 25:
            sample_images.append(image[0])
            sample_true.append(int(label[0]))
            sample_pred.append(int(predicted[0]))

    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    plot_predictions(sample_images, sample_true, sample_pred)
        
        
        
        
