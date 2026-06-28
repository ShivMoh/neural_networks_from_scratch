import numpy as np
def convert_ds_to_numpy(dataset_loader):
    labels, images = [], []
    for X, y in dataset_loader:
        images.append(X.numpy())
        labels.append(y.numpy())

    # labels = np.concatenate(labels, axis=0)
    # images = np.concatenate(images, axis=0)

    return labels, images
