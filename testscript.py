import sys
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader

from IdridConv import MyIdridNet
from read_dataset import MyIDRiDImageDataset
#import matplotlib.pyplot as plt
import torchvision.transforms as transforms



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
    net.load_state_dict(torch.load('saved_models/prvni.pt', map_location=device))
    net.eval()
    net.to(device)

    testing_data = MyIDRiDImageDataset(
        img_dir='dataset/A. Segmentation/A. Segmentation/1. Original Images/b. Testing Set',
        labels_img_dir='dataset/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/5. Optic Disc',
        resize=(256, 256))

    test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=False)


    for i, data in enumerate(test_dataloader, 55):
        # get the inputs
        inputs, labels = data

        outputs = net(inputs.to(device))


        out = outputs.squeeze()
        outcol = torch.zeros((3, outputs.size()[2], outputs.size()[3]))
        outcol[1, :, :] = outputs
        outcol[0, :, :] = labels
        imout = transforms.Resize((2848, 4288), antialias=True)(outcol)

        save_image(torch.round(imout), 'saved_images/'+str(i)+'_out.png')
        #torch.save(out, 'saved_images/'+str(i)+'_out.pt')
        print(str(i) + " saved")



        #break

print('End of test')



