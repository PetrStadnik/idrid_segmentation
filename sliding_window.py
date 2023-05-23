import torch
import sys
from torch import optim
from read_dataset import MyIDRiDImageDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':


    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=False)

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

    learning_rate = 0.00001
    momentum = 0.9
    win_size = 1024

    print("Learning rate: " + str(learning_rate) + " Momentum: " + str(momentum))

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    lossfc = torch.nn.CrossEntropyLoss()

    d = 0

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.000
        for i, data in enumerate(train_dataloader, 1):
            torch.cuda.empty_cache()
            inputs, labels = data
            for sw in range(11):
                for sh in range(20):
                    winput = inputs[:, :, sh*96:sh*96+win_size, sw*224:sw*224+win_size*2]
                    wlabel = labels[:, :, sh*96:sh*96+win_size, sw*224:sw*224+win_size*2]
                    optimizer.zero_grad()
                    outputs = model(winput.to(device))
                    loss = lossfc(outputs.squeeze(), wlabel.to(device).squeeze())
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    d += 1
                    with torch.no_grad():
                        if d % 10 == 0:
                            print(epoch + 1, i, '  loss:  ', float(running_loss/10))
                            """
                            label = labels.squeeze()
                            outputs = outputs.squeeze()
                            plt.imshow(outputs)
                            plt.show()
                            plt.imshow(label)
                            plt.show()
                            """
                            running_loss = 0.000


    print('Finished Training')
    torch.save(model.state_dict(), 'saved_models/swHard1.pt')
    print('Model saved!')
