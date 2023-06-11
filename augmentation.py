import sys
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader

from IdridConv import MyIdridNet
from read_dataset import MyIDRiDImageDataset
#import matplotlib.pyplot as plt
import torchvision.transforms as transforms

img_data = MyIDRiDImageDataset(
        img_dir='dataset/A. Segmentation/A. Segmentation/1. Original Images/a. Training Set',
        labels_img_dir='dataset/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates',
        resize=None,
        normalize=True)

train_dataloader = DataLoader(img_data, batch_size=1, shuffle=False)

id = 324
# Augmentation for Hard Exudates
for i, data in enumerate(train_dataloader, 1):
    inputs, labels = data

    hflip = transforms.RandomHorizontalFlip(p=1)

    for k in range(5):

        transformer = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=.02)
        transformed_img = hflip(transformer(inputs.squeeze()))
        save_image(transformed_img, 'augmented/original/' + str(id) + '_image.png')
        save_image(torch.round(hflip(labels)), 'augmented/hard_exudates_labels/' + str(id) + '_label.tif')
        id +=1


