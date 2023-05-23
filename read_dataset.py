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
    def __init__(self, img_dir, labels_img_dir, normalize=False, resize=None):
        self.labels_img_dir = labels_img_dir
        self.img_dir = img_dir
        self.all_images = os.listdir(self.img_dir)
        self.all_labels = os.listdir(self.labels_img_dir)
        self.normalize = normalize
        self.resize = resize

    def window(self, image, size=1024):
        ...

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
        if self.normalize:
            #image = transforms.Normalize(image.mean((1, 2)), image.std((1, 2)))(image)
            #print(torch.min(image, dim=(1,2)))
            #image = transforms.Normalize(image.min(dim=0), (image.max(dim=0) - image.min(dim=0)))(image)

            for i in range(3):
                image[i, :, :] = (image[i, :, :] - image[i,:,:].min()) / (torch.max(image[i, :,:]) - torch.min(image[i, :, :]))
        else:
            image = transform(image)

        if self.resize:
            image = transforms.Resize(self.resize, antialias=True)(image)
            imlabel = transforms.Resize(self.resize, antialias=True)(imlabel)

        myLabelsTransform = transforms.Compose(
            [transforms.Normalize(0, (imlabel.max() - imlabel.min()))])

        imlabel = torch.round(myLabelsTransform(imlabel))

        return image, imlabel


if __name__ == '__main__':
    training_data = MyIDRiDImageDataset(
        img_dir='dataset/A. Segmentation/A. Segmentation/1. Original Images/a. Training Set',
        labels_img_dir='dataset/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates',
        normalize=True)

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
