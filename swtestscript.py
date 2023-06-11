import sys
from torchvision.utils import save_image
import torch
from torch.utils.data import DataLoader
from read_dataset import MyIDRiDImageDataset
import matplotlib.pyplot as plt



t = 0.9
net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=False)


if __name__=='__main__':
    print("Hello in testing :-)")
    #net = MyIdridNet()

    win_size=1024

    #print(divide_mask.max())
    #print(divide_mask.min())



    if torch.cuda.is_available():
        dev = "cuda:" + sys.argv[1]
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(device)

    #net = MyIdridNet()
    net.load_state_dict(torch.load('saved_models/swHard1.pt', map_location=device))
    net.eval()
    net.to(device)

    divide_mask = torch.zeros((2848, 4288))
    divide_mask = divide_mask.to(device)
    for sw in range(11):
        for sh in range(20):
            divide_mask[sh * 96:sh * 96 + win_size, sw * 224:sw * 224 + win_size * 2] = \
                divide_mask[sh * 96:sh * 96 + win_size, sw * 224:sw * 224 + win_size * 2]+1


    plt.imshow(divide_mask)
    plt.show()


    testing_data = MyIDRiDImageDataset(
        img_dir='dataset/A. Segmentation/A. Segmentation/1. Original Images/b. Testing Set',
        labels_img_dir='dataset/A. Segmentation/A. Segmentation/2. All Segmentation Groundtruths/b. Testing Set/3. Hard Exudates',
        normalize=True)

    test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=True)

    total_green = 0
    total_wrong = 0



    for i, data in enumerate(test_dataloader, 55):

        outputs = torch.zeros((1,1,2848, 4288))
        outputs = outputs.to(device)
        inputs, labels = data
        labels = labels.to(device)
        for sw in range(11):
            for sh in range(20):
                winput = inputs[:, :, sh * 96:sh * 96 + win_size, sw * 224:sw * 224 + win_size * 2].to(device)
                with torch.no_grad():
                    outputs[:, :, sh * 96:sh * 96 + win_size, sw * 224:sw * 224 + win_size * 2] = \
                        outputs[:, :, sh * 96:sh * 96 + win_size, sw * 224:sw * 224 + win_size * 2] + net(winput.to(device))

        outputs = outputs/divide_mask

        outcol = torch.zeros((3, outputs.size()[2], outputs.size()[3]))
        outcol = outcol.to(device)
        outcol[0, :, :] = torch.where(labels < 0.5, torch.where(outputs >= t, 1, 0), 0)
        outcol[1, :, :] = torch.where(labels > 0.5, torch.where(outputs >= t, 1, 0), 0)
        outcol[2, :, :] = torch.where(outputs < t, torch.where(labels > 0.5, 1, 0), 0)

        error = 100*(1-(torch.sum(outcol[1, :, :])/torch.sum(outcol)))

        total_green += torch.sum(outcol[1, :, :])
        total_wrong += torch.sum(outcol)

        #outcol = transforms.Resize((2848, 4288), antialias=True)(outcol)

        save_image(torch.round(outcol), 'saved_images/swHard1/'+str(i)+'_out.png')
        #torch.save(out, 'saved_images/'+str(i)+'_out.pt')
        print(str(i) + " Error: " + str(error.item()) + "%")




        #break
    total_error = 100*(1-(total_green/total_wrong))
    print("Total error: ")
    print(total_error.item())

    print('End of test')



