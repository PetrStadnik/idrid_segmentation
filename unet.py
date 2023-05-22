import torch
import sys
from torch import optim
from read_dataset import MyIDRiDImageDataset
from torch.utils.data import Dataset, DataLoader

if __name__ == '__main__':


    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, pretrained=False)

    print("Hello to my U-net test :-)")

    if torch.cuda.is_available():
        dev = "cuda:" + sys.argv[1]
    else:
        dev = "cpu"

    device = torch.device(dev)
    print(device)
    model.to(device)

    training_data = MyIDRiDImageDataset(img_dir='data/training_set/',
                                        labels_img_dir='data/disk_labels/')

    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=False)

    #criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.000001, momentum=0.9)
    lossfc = torch.nn.CrossEntropyLoss()

    for epoch in range(1):  # loop over the dataset multiple times

        running_loss = 0.000
        for i, data in enumerate(train_dataloader, 0):
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
            #loss = torch.max(torch.abs(outputs-labels.to(device)))
            loss = lossfc(outputs, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 1 == 0:
                print(epoch + 1, i + 1, '  loss:  ', running_loss)
                running_loss = 0.000


    print('Finished Training')
    torch.save(model.state_dict(), 'saved_models/prvni.pt')
    print('Model saved!')
