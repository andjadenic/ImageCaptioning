import os
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
import skimage.io as io


class MiniCoco(Dataset):
    # MiniCOCO dataset with captions
    def __init__(self, csv_file, root_dir, transform=None):
        '''
        Arguments:
        :param csv_file: Path to the csv file with captions
        :param root_dir: Directory with all the images
        :param transform (callable, optional): Optional transform to be applied
        '''
        self.frame = pd.read_csv(csv_file)  # Panda's DataFrame with all captions
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.frame.iloc[idx, 0]
        img_path = f'data/miniCOCO/train/{img_name}'
        image = io.imread(img_path)
        captions = self.frame.iloc[idx, 1:]  # Pandas Series

        sample = {'image': image, 'captions': captions.to_dict()}
        if self.transform:
            sample = self.transform(sample)
        return sample



if __name__ == '__main__':

    csv_file = f'data/miniCOCO/train_captions.csv'
    root_dir = f'data/miniCOCO/train'

    # Make torch's Dataset out of miniCOCO train dataset
    train_dataset = MiniCoco(csv_file, root_dir)

    # Define transformation pipeline
    data_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # Randomly crops the image to a size (224,224,3)
        transforms.RandomHorizontalFlip(),  # Randomly flips the image horizontally with a 50% probability
        transforms.ToTensor(),  # Converts the image from a PIL image or NumPy array to a PyTorch tensor
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])  # Normalizes the pixel values using the
    # mean and standard deviation of the ImageNet dataset.
    # The pre-trained ResNet model expects input images to be normalized this way.

    # Make torch's DataLoader
    dataloader = DataLoader(train_dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=0)

