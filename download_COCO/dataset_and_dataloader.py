# https://docs.pytorch.org/vision/stable/datasets.html
# https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CocoCaptions.html#torchvision.datasets.CocoCaptions


import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import *
import time


# For captions, you typically need to build a vocabulary and tokenize them.
# The target_transform here would convert a list of strings (captions) into numerical representations.
# This example just passes the raw captions.
def target_transform_fn(captions):    # how target_transform should transform a sentance???? just indexed and padded sentance?
    # use built Vocabulary to do this
    # In a real scenario, you'd process these captions:
    # 1. Tokenize them (e.g., using NLTK or SpaCy)
    # 2. Build a vocabulary mapping words to numerical IDs
    # 3. Convert captions to sequences of numerical IDs
    # 4. Pad sequences to a fixed length

    return captions # For demonstration, return as-is


if __name__ == '__main__':
    # Define preprocessing transformations for the images
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    # Use built-in CocoCaptions class to load the train dataset
    train_data = datasets.CocoCaptions(
        root=train_data_path,
        annFile=train_annFile,
        transform=transform,
        target_transform=target_transform_fn
    )

    # Load the test dataset
    test_data = datasets.CocoCaptions(
        root=test_data_path,
        annFile=test_annFile,
        transform=transform,
        target_transform=target_transform_fn
    )

    #print('Number of train samples: ', len(train_data))
    #print('Number of test samples: ', len(test_data))

    # Create DataLoaders for batching and shuffling
    train_loader = DataLoader(train_data,
                              batch_size=Nb,
                              shuffle=True,
                              num_workers=4)  # num_workers determines how many subprocesses to use for data loading


    print("Loading with num_workers = 0:")
    start_time = time.time()
    loader_0 = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=0)
    for i, (data, target) in enumerate(loader_0):
        if i == 100: # Just iterate a few batches
            break
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds\n")

    # DataLoader with multiple workers
    print("Loading with num_workers = 4:")
    start_time = time.time()
    loader_4 = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=4)
    for i, (data, target) in enumerate(loader_4):
        if i == 100: # Just iterate a few batches
            break
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
