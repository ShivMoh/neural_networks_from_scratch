import numpy as np
def convert_ds_to_numpy(dataset_loader, no_batch=True):
    labels, images = [], []
    for X, y in dataset_loader:
        images.append(X.numpy())
        labels.append(y.numpy())


    # So what this is doing basically is removing batches. so instead of X batches of 64
    # we put it in one big numpy array all at once
    if no_batch:
        labels = np.concatenate(labels, axis=0)
        images = np.concatenate(images, axis=0)

    return labels, images
