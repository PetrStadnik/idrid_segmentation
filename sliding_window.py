import torch
import sys
from torch import optim
from read_dataset import MyIDRiDImageDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':


    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=64, pretrained=False)

    print("Hello to my sliding window :-)")

    if torch.cuda.is_available():
        dev = "cuda:" + sys.argv[1]
    else:
        dev = "cpu"

    device = torch.device(dev)
    print(device)
    model.to(device)

    training_data = MyIDRiDImageDataset(img_dir='dataset/A. Segmentation/A. Segmentation/1. Original Images/a. Training Set',
                                        labels_img_dir='dataset/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/3. Hard Exudates',
                                        resize=None,
                                        normalize=True)

    learning_rate = 0.0000001
    momentum = 0.9
    win_size = 1024

    print("Learning rate: " + str(learning_rate) + " Momentum: " + str(momentum))

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    lossfc = torch.nn.CrossEntropyLoss()

    d = 0

    for epoch in range(8):  # loop over the dataset multiple times

        running_loss = 0.000
        for i, data in enumerate(train_dataloader, 1):
            torch.cuda.empty_cache()
            inputs, labels = data
            for sw in range(4):
                for sh in range(3):
                    winput = inputs[:, :, 20*sh*32:20*sh*32+win_size, 11*sw*56:11*sw*56+win_size*2]
                    wlabel = labels[:, :, 20*sh*32:20*sh*32+win_size, 11*sw*56:11*sw*56+win_size*2]
                    optimizer.zero_grad()
                    outputs = model(winput.to(device))
                    loss = lossfc(outputs.squeeze(), wlabel.to(device).squeeze())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    d += 1
                    with torch.no_grad():
                        if d % 10 == 0:
                            print(epoch + 1, i, '  loss:  ', float(running_loss/100))

                            label = labels.squeeze()
                            outputs = outputs.squeeze()
                            plt.imshow(outputs)
                            plt.show()
                            plt.imshow(label)
                            plt.show()

                            running_loss = 0.000


    print('Finished Training')
    torch.save(model.state_dict(), 'saved_models/swHard1.pt')
    print('Model saved!')
