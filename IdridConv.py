import sys

import torch                     # for all things PyTorch
import torch.nn as nn            # for torch.nn.Module, the parent object for PyTorch models
import torch.nn.functional as F  # for the activation function
from torch import optim
from read_dataset import MyIDRiDImageDataset
from torch.utils.data import Dataset, DataLoader

class MyIdridNet(nn.Module):
    def __init__(self):
        super(MyIdridNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding=2,)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.softmax(x, dim=1)
        return x




if __name__=='__main__':

    print("Hello to my net :-)")
    net = MyIdridNet()

    if torch.cuda.is_available():
        dev = "cuda:"+sys.argv[1]
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(device)
    net.to(device)

    training_data = MyIDRiDImageDataset(
        img_dir='dataset/A. Segmentation/A. Segmentation/1. Original Images/a. Training Set',
        labels_img_dir='dataset/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc',
        resize=(256, 256))

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.000001, momentum=0.9)

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.000
        for i, data in enumerate(train_dataloader, 0):
            # get the inputs
            inputs, labels = data
            #print(f"Feature batch shape: {inputs.size()}")
            #print(f"Labels batch shape: {labels.size()}")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            #print(f"Got output batch with shape: {outputs.size()}")
            #loss = torch.max(torch.abs(outputs-labels.to(device)))
            loss = criterion(outputs.squeeze(), labels.to(device).squeeze())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:
                print(epoch + 1, i + 1,'  loss:  ', running_loss)
                running_loss = 0.000

    print('Finished Training')
    torch.save(net.state_dict(), 'saved_models/prvni.pt')
    print('Model saved!')
