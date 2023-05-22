import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function

import numpy as np
import matplotlib.pyplot as plt
from torch import optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from torchvision.transforms import ToTensor


class MyIDRiDImageDataset(Dataset):
    def __init__(self, img_dir, labels_img_dir, transform=None, target_transform=None):
        self.labels_img_dir = labels_img_dir
        self.img_dir = img_dir
        self.all_images = os.listdir(self.img_dir)
        self.all_labels = os.listdir(self.labels_img_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.all_images[idx])
        image = read_image(img_path).float()
        imlabel = Image.open(os.path.join(self.labels_img_dir, self.all_labels[idx]))
        imlabel = ToTensor()(imlabel)
        imlabel.float()

        transform = transforms.Compose(
            [transforms.Normalize((0.0, 0.0, 0.0), (255, 255, 255))])
        myLabelsTransform = transforms.Compose(
            [transforms.Normalize(0.0, 0.0039215)])

        image = transform(image)
        imlabel = myLabelsTransform(imlabel)

        return image, imlabel


if __name__ == '__main__':

    training_data = MyIDRiDImageDataset(img_dir='data/training_set/',
                                        labels_img_dir='data/disk_labels/')

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

    print(training_data)

    train_features, train_labels = next(iter(train_dataloader))
    print(train_labels.max())
    print(train_labels.min())
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    print(train_features.max())
    print(train_features.min())
    img = train_features.squeeze()
    label = train_labels.squeeze()
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    plt.imshow(label)
    plt.show()
