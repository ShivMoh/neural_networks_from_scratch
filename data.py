def convert_ds_to_numpy(dataset_loader):
    labels, images = [], []
    for X, y in dataset_loader:
        images.append(X.numpy())
        labels.append(y.numpy())

    return labels, images
