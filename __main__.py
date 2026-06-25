from nn.nn import Linear, Flatten
from nn.activation import ReLU
from nn.loss import Cross_Entropy_Loss
from nn.optimizer import SGD
from nn.data import convert_ds_to_numpy
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


if __name__ == "__main__":
    print("I am testing to see if this works")

    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # Batch size 1 for prediction

    images, labels = convert_ds_to_numpy(train_loader)


    for image in images:
        print("Image", image.shape)
    
        flatten = Flatten()
        linear = Linear(64, 10)
        x = flatten.forward(image)
        x = linear.forward(x)

        print(x)

        break
