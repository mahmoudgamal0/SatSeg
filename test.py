import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
from torchgeo.datasets import LoveDA
from torch.utils.data import DataLoader, SequentialSampler
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from model import U2NET
from pre import augment, class_to_onehot, flip_axis, onehot_to_color, gen_handles

from model import U2NET
import cv2
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
from torchgeo.datasets import LoveDA

HANDLES = gen_handles()

def save_image(name, image, mask):
    f, ax = plt.subplots(2, 1, figsize=(10, 10), squeeze=True)
    ax[0].imshow(flip_axis(image))
    ax[1].imshow(onehot_to_color(mask), alpha=0.3)
    ax[1].legend(handles=HANDLES)
    plt.savefig(f'./data/results/{name}.png', facecolor= 'w', transparent= False, bbox_inches= 'tight', dpi= 100)
    plt.close()
    
def main():

    # --------- 1. get image path and name ---------

    transforms = augment(256, 256, for_mask=False)
    test_dataset = LoveDA(root="./data", split='test', transforms=transforms)
    
    test_dataloader = DataLoader(test_dataset, batch_size = 1)
   

    # --------- 2. model define ---------
    net = U2NET(3, 8)
    model_dir = "./data/saved_models/u2net_ce_itr_78000_train_5.179635_tar_0.726550_acc_70.420000.pth"

    
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 3. inference for each image ---------
    for i, sample in enumerate(test_dataloader):

        image = sample['image']
        image = image.type(torch.FloatTensor)

        if torch.cuda.is_available():
            image = Variable(image.cuda(), requires_grad=False)
        else:
            image = Variable(image)
        
        d0,d1,d2,d3,d4,d5,d6 = net(image)

        jpg_image = image[0].type(torch.IntTensor).detach().cpu().numpy()
        
        jpg_mask = d0[0]
        jpg_mask[0] = 0
        jpg_mask = torch.argmax(jpg_mask, axis=0).detach().cpu().numpy()

        save_image(i, jpg_image, jpg_mask)
        print(f'saved sample {i}')
        # normalization
        # pred = d1[:,0,:,:]
        # pred = normPRED(pred)

        # save results to test_results folder
        # if not os.path.exists(prediction_dir):
        #     os.makedirs(prediction_dir, exist_ok=True)
        # save_output(img_name_list[i_test],pred,prediction_dir)

        del d0,d1,d2,d3,d4,d5,d6

if __name__ == "__main__":
    main()
