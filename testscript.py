import sys
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader

from IdridConv import MyIdridNet
from read_dataset import MyIDRiDImageDataset
#import matplotlib.pyplot as plt
import torchvision.transforms as transforms


t = 0.9
net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=False)


if __name__=='__main__':
    print("Hello in testing :-)")
    #net = MyIdridNet()

    if torch.cuda.is_available():
        dev = "cuda:" + sys.argv[1]
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(device)
    net.to(device)

    #net = MyIdridNet()
    net.load_state_dict(torch.load('saved_models/unet4.pt', map_location=device))
    net.eval()
    net.to(device)


    testing_data = MyIDRiDImageDataset(
        img_dir='dataset/A. Segmentation/A. Segmentation/1. Original Images/b. Testing Set',
        labels_img_dir='dataset/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/5. Optic Disc',
        resize=(1024, 1024),
        normalize=True)

    test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=False)

    total_green = 0
    total_wrong = 0

    for i, data in enumerate(test_dataloader, 55):
        # get the inputs
        inputs, labels = data
        with torch.no_grad():
            outputs = net(inputs.to(device))

        outcol = torch.zeros((3, outputs.size()[2], outputs.size()[3]))
        outcol[0, :, :] = torch.where(labels < 0.5, torch.where(outputs >= t, 1, 0), 0)
        outcol[1, :, :] = torch.where(labels > 0.5, torch.where(outputs >= t, 1, 0), 0)
        outcol[2, :, :] = torch.where(outputs < t, torch.where(labels > 0.5, 1, 0), 0)

        error = 100*(1-(torch.sum(outcol[1, :, :])/torch.sum(outcol)))

        total_green += torch.sum(outcol[1, :, :])
        total_wrong += torch.sum(outcol)

        #outcol = transforms.Resize((2848, 4288), antialias=True)(outcol)

        save_image(torch.round(outcol), 'saved_images/unet4/'+str(i)+'_out.png')
        #torch.save(out, 'saved_images/'+str(i)+'_out.pt')
        print(str(i) + " Error: " + str(error.item()) + "%")



        #break
    total_error = 100*(1-(total_green/total_wrong))
    print("Total error: ")
    print(total_error.item())

    print('End of test')



