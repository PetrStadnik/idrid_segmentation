import sys
import torch
from torch.utils.data import DataLoader
from read_dataset import MyIDRiDImageDataset
import matplotlib.pyplot as plt




if __name__=='__main__':
    print("Hello in showing :-)")

    testing_data = MyIDRiDImageDataset(img_dir='data/testing_set/',
                                        labels_img_dir='data/testing_disk_labels/')

    test_dataloader = DataLoader(testing_data, batch_size=1, shuffle=False)

    for i, data in enumerate(test_dataloader, 55):
        inputs, labels = data


        print(f"Feature batch shape: {inputs.size()}")
        #print(f"Labels batch shape: {.size()}")
        img = inputs.squeeze()
        lab = labels.squeeze()
        out = torch.load('saved_images/' + str(i) + '_out.pt', map_location=torch.device('cpu'))
        print(out.max())
        print(out.min())

        plt.imshow(img.permute(1,2,0).detach().numpy())
        plt.show()
        plt.imshow(lab)
        plt.show()
        plt.imshow(out.detach().numpy())
        plt.show()
        break

print('End of showing')



