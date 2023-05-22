import torch
import sys
from torch import optim
from read_dataset import MyIDRiDImageDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

if __name__ == '__main__':


    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=False)

    print("Hello to my sliding_window :-)")

    if torch.cuda.is_available():
        dev = "cuda:" + sys.argv[1]
    else:
        dev = "cpu"

    device = torch.device(dev)
    print(device)
    model.to(device)

    training_data = MyIDRiDImageDataset(img_dir='dataset/A. Segmentation/A. Segmentation/1. Original Images/a. Training Set',
                                        labels_img_dir='dataset/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/a. Training Set/5. Optic Disc',
                                        resize=(1024, 1024))

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    lossfc = torch.nn.CrossEntropyLoss()

    for epoch in range(3):  # loop over the dataset multiple times

        running_loss = 0.000
        for i, data in enumerate(train_dataloader, 0):
            print(i)
            print("----")
            torch.cuda.empty_cache()
            # get the inputs
            inputs, labels = data
            #print(f"Feature batch shape: {inputs.size()}")
            #print(f"Labels batch shape: {labels.size()}")

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            #print(f"Got output batch with shape: {outputs.size()}")
            #loss = torch.sum(torch.pow(outputs-labels.to(device),2))
            loss = lossfc(outputs.squeeze(), labels.to(device).squeeze())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            with torch.no_grad():
                if i % 10 == 0:
                    print(epoch + 1, i + 1, '  loss:  ', float(running_loss))

                    label = labels.squeeze()
                    outputs = outputs.squeeze()
                    plt.imshow(outputs)
                    plt.show()
                    plt.imshow(label)
                    plt.show()
                    running_loss = 0.000


    print('Finished Training')
    torch.save(model.state_dict(), 'saved_models/unet.pt')
    print('Model saved!')
