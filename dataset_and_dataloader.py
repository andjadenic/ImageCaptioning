# https://docs.pytorch.org/vision/stable/datasets.html
# https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CocoCaptions.html#torchvision.datasets.CocoCaptions


import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import *


if __name__ == '__main__':
    # Define preprocessing transformations for the images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Built-in CocoCaptions class to load the train dataset
    train_data = datasets.CocoCaptions(
        root=train_data_path,
        annFile=train_annFile,
        transform=transform,  # preprocess images
    )

    # Load the test dataset
    test_data = datasets.CocoCaptions(
        root=test_data_path,
        annFile=test_annFile,
        transform=transform
    )


    # Create DataLoaders for batching and shuffling
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)  # num_workers determines how many subprocesses to use for data loading


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    f


    train_data = datasets.CocoCaptions(
        root=train_data_path,
        annFile=train_annFile,
        transform=transform  # preprocess images
    )

    for i, (images, captions) in enumerate(train_loader):
        print(captions, '\n')
        if i == 5:
            break