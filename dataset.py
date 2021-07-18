import os, sys
import numpy as np
import cv2
from PIL import Image
import json


import torch
from torch.utils.data import Dataset #, Dataloader
import torchvision
import torchvision.transforms as transforms

def json_parse(image_dir, json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    file_name_list = []
    anno = []
    for image_name in data:
        file_name_list.append(os.path.join(image_dir, image_name))
        anno.append(data[image_name])

    return file_name_list, anno

class CarotidSet(Dataset):
    def __init__(self, iamge_root, json_path, transform=None, flip=True, rotate=True, crop=True):
        self.transform = transform
        
        self.image_list, self.anno = json_parse(iamge_root, json_path)
   
        if flip:
            self.flip = transforms.RandomHorizontalFlip(p=.5)
        else:
            self.flip = None
        if rotate:
            self.rotate = transforms.RandomRotation(degrees=(0,180))
        else:
            self.rotate = None 
        self.resize = transforms.Resize(size=(512,512))

        

   


    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        #self.image_list, self.anno = json_parse(root)
        return self.image_list[index], self.anno[index]


        #img = cv2.imread(self.samples[index])
        img = Image.open(self.samples[index])
        img = self.resize(img) # resize to 512 x 512
        
        # random rotate
        if self.rotate:
            if np.random.rand() < 0.5:
                img = self.rotate(img)
        
        # whitening
        if self.transform:
            img = self.transform(img)

        # random horizontal flip
        if self.flip:
            img = self.flip(img)
        

        return img
        

def tensor2numpy(img):
    img = ((img*0.5) + 0.5).clamp(0.0, 1.0) # -1~1 -> 0 ~ 1
    # 0 ~ 1 -> 0 ~ 255  
    np_img = (img.cpu().detach() * 255.).numpy().astype(np.uint8)
    # C x H x W -> H x W x C
    np_img = np_img.transpose(1,2,0)[:,:,::-1]
    return np_img

if __name__ == '__main__':
    junkfood_transform = transforms.Compose(
        [
            transforms.ToTensor(), # 0~255 -> 0~1
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # 0~1 -> -1~1
        ])
    train_set = CarotidSet('./image', './gTruth.json', transform=junkfood_transform, 
                            flip=False, rotate=True, crop=False)


    for idx, data in enumerate(train_set):
        file_name, annotation = data
        print(file_name)
        print(annotation)
        exit(1)
        
        
       
