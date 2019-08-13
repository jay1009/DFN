import os
import sys
import torch
from torch.utils import data
from PIL import Image
import numpy as np
from torchvision import transforms
import operator

# palette = {
#     'background':[0, 0, 0],
#     'aeroplane':[128, 0, 0],
#     'bicycle':[0, 128, 0],
#     'bird':[128, 128, 0],
#     'boat':[0, 0, 128],
#     'bottle':[128, 0, 128],
#     'bus':[0, 128, 128],
#     'car':[128, 128, 128],
#     'cat':[64, 0, 0],
#     'chair':[192, 0, 0],
#     'cow':[64, 128, 0],
#     'diningtable':[192, 128, 0],
#     'dog':[64, 0, 128],
#     'horse':[192, 0, 128],
#     'motorbike':[64, 128, 128],
#     'person':[192, 128, 128],
#     'pottedplant':[0, 64, 0],
#     'sheep':[128, 64, 0],
#     'sofa':[0, 192, 0],
#     'train':[128, 192, 0],
#     'tvmonitor':[0, 64, 128],
#     'void':[224, 224, 192]}


palette = [[0, 0, 0],
           [255, 255, 255]]



def to_mask(x):
    """
    input : (None, 3, H, W) - RGB value
    output : (None, H, W) - Label value
    """
    np_x = np.array(x)
    H, W, C = np_x.shape
    flatten_np_x = np_x.reshape(-1, C)
    #print(flatten_np_x)   # 65536, 3
    empty = np.zeros_like(flatten_np_x)[:, 0]
    #print(empty.shape)  # (65536,)
    for i, it in enumerate(flatten_np_x):
        # it = (R, G, B)
        #print("it: ", it)
        #print("list(it): ", list(it))
        # Remove void part as background
        if it[0]<=128 and it[1]<=128 and it[2]<=128:
            empty[i] = 0
        else:
            empty[i] = 1

    mask = empty.reshape(H, W, 1).transpose(2, 0, 1)
    return torch.from_numpy(mask).squeeze().long()

def to_rgb(xs):
    """
    input : (None, H, W) - Label value
    output : (None, 3, H, W) - RGB value
    """
    rgbs = np.zeros((xs.size(0), xs.size(1), xs.size(2), 3))
    for i, x in enumerate(xs):
        np_x = np.array(x)
        H, W = np_x.shape
        flatten_np_x = np_x.reshape(-1)
        expand_np_x = np_x.reshape(-1, 1).repeat(3, axis=-1) # (H, W, 3)
        for j in range(2):
            expand_np_x[np.where(flatten_np_x == j)] = palette[j]
        rgbs[i] = expand_np_x.reshape(H, W, 3)
    rgbs = rgbs.transpose(0, 1, 2, 3)
    return rgbs

def test_path(root):
    root = '/home/cc105u/Segmentation-Pytorch/train/CVC-EndoSceneStill/CVC-612/'
    test_items = []
    img_path = os.path.join(root, 'bbdd/')
    mask_path = os.path.join(root, 'gtpolyp/')

    test_data_list = [l.strip('\n') for l in open(os.path.join(root, 'list/', 'test.txt')).readlines()]
    for it in test_data_list:
        item = (os.path.join(img_path, it + '.bmp'), os.path.join(mask_path, it + '.tif'))
        test_items.append(item)
    return test_items

def make_path(root):
    root = '/home/cc105u/Segmentation-Pytorch/train/CVC-EndoSceneStill/CVC-612/'
    train_items = []
    val_items = []

    img_path = os.path.join(root, 'bbdd/')
    mask_path = os.path.join(root, 'gtpolyp/')
    train_data_list = [l.strip('\n') for l in open(os.path.join(root, 'list/', 'train.txt')).readlines()]
    val_data_list = [l.strip('\n') for l in open(os.path.join(root, 'list/', 'val.txt')).readlines()]

    for it in train_data_list:
        item = (os.path.join(img_path, it + '.bmp'), os.path.join(mask_path, it + '.tif'))
        train_items.append(item)

    for it in val_data_list:
        item = (os.path.join(img_path, it + '.bmp'), os.path.join(mask_path, it + '.tif'))
        val_items.append(item)

    return train_items, val_items



class VOC(data.Dataset):

    def __init__(self, root, image_size, dataset_type, transform=None, target_transform=to_mask):
        """
        root - parent of data file
        dataset_type - ['train', 'val']
        """
        assert dataset_type in ['train', 'val', 'test'], 'dataset_type should be in train/val/test'
        self.train_items, self.val_items = make_path(root)
        self.h_image_size, self.w_image_size = image_size[0], image_size[1]
        self.dataset_type = dataset_type
        self.transform = transform
        self.target_transform = target_transform
        self.test_items = test_path(root)

    def __getitem__(self, i):
        if self.dataset_type == 'train':
            index = np.random.choice(len(self.train_items), 1)[0]
            name = self.train_items[index]
        elif self.dataset_type == 'val':
            index = np.random.choice(len(self.val_items), 1)[0]
            name = self.val_items[index]
        elif self.dataset_type == 'test':
            index = np.random.choice(len(self.val_items), 1)[0]
            name = self.val_items[index]

        image = Image.open(name[0]).convert('RGB')

        mask = Image.open(name[1]).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            mask = transforms.Pad(10)(mask)
            mask = transforms.CenterCrop((self.h_image_size, self.w_image_size))(mask)
            mask = self.target_transform(mask)

        return image, mask

    def __len__(self):
        if self.dataset_type == 'train':
            return len(self.train_items)
        elif self.dataset_type == 'val':
            return len(self.val_items)
        elif self.dataset_type == 'test':
            return len(self.test_items)

if __name__ == "__main__":
    print(sys.path[0])
    transform = transforms.Compose([transforms.Resize(128), transforms.ToTensor()])
    data_set = VOC(root="/home/cc105u/Segmentation-Pytorch/train/CVC-EndoSceneStill/CVC-612/", image_size=128, dataset_type='train', transform=transform, target_transform=to_mask)
    # for data in data_set:
    #     print(np.array(data[0]).shape, np.array(data[1]).shape)
    # np.set_printoptions(threshold=np.nan)
    print(data_set[0][1].type())
    # print(data_set[0][1])
