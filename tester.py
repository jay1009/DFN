import os
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.voc import VOC
from trainer import Trainer
import torch
from torch import optim
import torch.nn as nn
from models import FCN, unet, pspnet, dfn
from datasets.voc import to_mask
from datasets.voc import to_rgb
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import time
from datetime import timedelta
from PIL import Image

transform = transforms.Compose([
                                transforms.Pad(10),
                                transforms.CenterCrop(256),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])])

root = '/home/cc105u/Segmentation-Pytorch/train/'
test_data_set = VOC(root=root,
                   image_size=(256,256),
                   dataset_type='test',
                   transform=transform)
test_data_loader = DataLoader(test_data_set,
                             batch_size=4,
                             shuffle=False) # For make samples out of various models, shuffle=False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#model = unet.UNet(num_classes=2, in_dim=3, conv_dim=64)
model = dfn.SmoothNet(num_classes=2,
                      h_image_size=256,
                      w_image_size=256)
model = model.to(device)
model.load_state_dict(torch.load('Cvc-unet.pth'))
model.eval()
test_start_time = time.time()
data_iter = iter(test_data_loader)
max_iter = len(test_data_loader)
n_iter = 0
input, target = next(data_iter)
input_var = input.clone().to(device)
target_var = target.to(device)

#print('input_var=',input_var[0])
output = model(input_var)
#print('output=',output)
output_label = torch.argmax(output, dim=1)
#print('output_label=',output_label)
#print('output_label.shape=',output_label.shape)
output_rgb = to_rgb(output_label[0:4].cuda())
#print('output_rgb.shape=',output_rgb.shape)
ndarray_convert_img = Image.fromarray((output_rgb[0]).astype(np.uint8))
ndarray_convert_img.save('output.bmp')

"""
from PIL import Image
img = Image.open("82.bmp").convert('RGB')
input_t = transform(img)
input_var = input_t.clone().to('cuda:0')
print('input_var.shape',input_var.shape)
mask = Image.open("82.tif").convert('RGB')
mask = transforms.Pad(10)(mask)
mask = transforms.CenterCrop(256)(mask)
mask = to_mask(mask)

#batch_t = torch.unsqueeze(input_t, 0)
target_var = mask.to('cuda:0')

print('target_var.shape',target_var.shape)
output = model(input_var)
print('output.shape',output.shape)
output_label = torch.argmax(outout, dim=1)
"""
acc=0.0
TP = 0.0
FNP = 0.0
total=4*256*256
for i in range(4):
    for j in range(256):
        for k in range(256):
            if output_label[i,j,k] == 1 and target_var[i,j,k] == 1:
                TP += 1
            elif output_label[i,j,k] != target_var[i,j,k]:
                FNP += 1
print('TP:',TP)
print('FNP:',FNP)
acc = TP/(2*TP + FNP)
print("The meanIoU is ",acc,"%")
