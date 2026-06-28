from nn.nn import Linear, Flatten
from nn.activation import ReLU, Softmax
from nn.loss import Cross_Entropy_Loss, Activation_Softmax_Loss_CategoricalCrossEntropy
from nn.optimizer import SGD
from nn.data import convert_ds_to_numpy
from nn.plot import plot_loss, plot_predictions
from nn.convolution import Conv2D
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

    # labels_c = np.concatenate(labels, axis=0)
    # images_c = np.concatenate(images, axis=0)
    
    # print("images", images_c.shape)
    # print("labels", labels_c.shape)

    flatten = Flatten()
    linear_1 = Linear(729, 32)
    linear_2 = Linear(32, 10)
    relu = ReLU()
    optimizer = SGD(1)
    softmax_loss = Activation_Softmax_Loss_CategoricalCrossEntropy()
    conv2d_1 = Conv2D(kernel_size=(2,2))

    # loss = Cross_Entropy_Loss()
    # softmax = Softmax()
    
    losses = []

    for label, image in zip(labels, images):
        # print("Image", image.shape)
        # print("class label", label)
        # print(image.shape)
        transformed_images = []
        for im in image:
            img_2d = im.reshape(28,28)
            x = conv2d_1.forward(img_2d)
            transformed_images.append(x)
        
        x = flatten.forward(np.array(transformed_images))

        x = linear_1.forward(x)
        
        x = relu.forward(x)
        
        # softmax_output = softmax.forward(x)
        # loss = loss.compute(softmax_output, label)
        x = linear_2.forward(x)
        activation_loss = softmax_loss.forward(x, label)
        losses.append(activation_loss)
        
        dvalues = softmax_loss.backward(softmax_loss.output, label)
        dvalues = linear_2.backward(dvalues)
        dvalues = relu.backward(dvalues)
        dvalues = linear_1.backward(dvalues)
        dvalues = flatten.backward(dvalues)

        d_conv_outputs = dvalues

        for i, im in enumerate(image):
            conv2d_1.backward(d_conv_outputs[i])
            
        # print("before updates", linear.weights)
        # print("before updates", linear.biases)

        optimizer.update_params(linear_1, 1)
        optimizer.update_params(linear_2, 1)
        
        # print("after updates", linear.weights)
        # print("after updates", linear.biases)
        
        # dvalues = loss.backward(x, label)
    
    plot_loss(losses)    
    
    test_labels, test_images = convert_ds_to_numpy(test_loader)

    correct = 0
    total = 0
    sample_images, sample_true, sample_pred = [], [], []

    for label, image in zip(test_labels, test_images):
        transformed_images = []
        for im in image:
            img_2d = im.reshape(28, 28)
            x = conv2d_1.forward(img_2d)
            transformed_images.append(x)

        x = flatten.forward(np.array(transformed_images))
        x = linear_1.forward(x)
        x = relu.forward(x)
        x = linear_2.forward(x)
        probs = softmax_loss.activation.forward(x)

        predicted = np.argmax(probs, axis=1)

        if predicted[0] == label[0]:
            correct += 1
        total += 1

        if len(sample_images) < 25:
            sample_images.append(image[0])
            sample_true.append(int(label[0]))
            sample_pred.append(int(predicted[0]))

    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    plot_predictions(sample_images, sample_true, sample_pred)
        
        
        
        
