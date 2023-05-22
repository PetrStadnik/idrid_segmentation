import sys

import torch
from torch.utils.data import DataLoader

from IdridConv import MyIdridNet
from read_dataset import MyIDRiDImageDataset
#import matplotlib.pyplot as plt






if __name__=='__main__':
    print("Hello in testing :-)")
    net = MyIdridNet()

    if torch.cuda.is_available():
        dev = "cuda:" + sys.argv[1]
    else:
        dev = "cpu"
    device = torch.device(dev)
    print(device)
    net.to(device)

    net = MyIdridNet()
    net.load_state_dict(torch.load('saved_models/prvni.pt', map_location=device))
    net.eval()
    net.to(device)

    testing_data = MyIDRiDImageDataset(img_dir='data/testing_set/',
                                        labels_img_dir='data/testing_disk_labels/')

    test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=False)


    for i, data in enumerate(test_dataloader, 55):
        # get the inputs
        inputs, labels = data

        outputs = net(inputs.to(device))


        out = outputs.squeeze()
        print(i)
        torch.save(out, 'saved_images/'+str(i)+'_out.pt')
        print("saved")



        break

print('End of test')



