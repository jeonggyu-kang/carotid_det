import os, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw
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
    def __init__(self, image_root, json_path, transform=None, 
                        flip=True, rotation=True, translation=True):
        self.transform = transform        
        self.image_list, self.anno = json_parse(image_root, json_path)
                
        '''
        -	Augmentation
        	Randomly flip images
        	Randomly X/Y translation (-12 ~ 12 pixels)
        	Randomly rotation (-10︒ ~ 10︒).
        '''
        if flip:
            self.h_flip = transforms.RandomHorizontalFlip(p=.5)
            self.v_flip = transforms.RandomVerticalFlip(p=.5)
        else:
            self.h_flip = None
            self.v_flip = None
        
        if rotation:
            self.rotation = True
        else:
            self.rotation = None 

        if translation:
            #self.translation = transforms.
            self.translation = True
        else:
            self.translation = None 

        # TODO : roi resize 제안 내용 (adaptively determine height)
        self.resize = transforms.Resize(size=(128,128*3))
        #self.resize = transforms.Resize(size=(128,128))
        
    def __len__(self):
        return len(self.image_list)
        

    def make_gt(self, img_size, pt_list_x, pt_list_y):
        gt  = Image.new( mode = "L", size = img_size )
        # draw lines (LI & MA)
        draw = ImageDraw.Draw(gt)
        prev_x = pt_list_x[0] 
        prev_y = pt_list_y[0] 
        for (x, y) in zip(pt_list_x, pt_list_y):
            draw.line((prev_x, prev_y) + (x,y), fill=255)
            prev_x = x
            prev_y = y
        return gt
    
    def __getitem__(self, index):
        img = Image.open(self.image_list[index])

        #img = cv2.imread(self.image_list[index])
        roi = self.anno[index]['roi']
        li_x = self.anno[index]['li']['x']
        li_y = self.anno[index]['li']['y']
        ma_x = self.anno[index]['ma']['x']
        ma_y = self.anno[index]['ma']['y']
        
        # make LI & MA gt
        img_li  = self.make_gt(img.size, li_x, li_y)
        img_ma  = self.make_gt(img.size, ma_x, ma_y)

        # [1] translation
        if self.translation:
            x_translation = np.random.randint(-12,13)
            y_translation = np.random.randint(-12,13)

            roi[0] += x_translation
            roi[1] += y_translation
            roi[2] += x_translation
            roi[3] += y_translation

        # [2] rotation

        if self.rotation and np.random.randint(2) == 0:
            degree = np.random.randint(-10, 11)
            img    = transforms.functional.affine(img, degree, [0,0], 1.0, 0.0)
            img_li = transforms.functional.affine(img_li, degree, [0,0], 1.0, 0.0)
            img_ma = transforms.functional.affine(img_ma, degree, [0,0], 1.0, 0.0)

 



        img_roi = img.crop((roi[0], roi[1], roi[0] + roi[2], roi[1]+roi[3]))
        img_li  = img_li.crop((roi[0], roi[1], roi[0] + roi[2], roi[1]+roi[3]))
        img_ma  = img_ma.crop((roi[0], roi[1], roi[0] + roi[2], roi[1]+roi[3]))
 
        img_roi = self.resize(img_roi)
        img_li  = self.resize(img_li)
        img_ma  = self.resize(img_ma)

        if self.h_flip:
            img_roi = self.h_flip(img_roi)
            img_li  = self.h_flip(img_li)
            img_ma  = self.h_flip(img_ma)
        
        threshold = 128
        img_li = img_li.point(lambda p: p > threshold and 255)  
        img_ma = img_ma.point(lambda p: p > threshold and 255)  

        # TODO
        # apply data augmentations used in
        # K.L et al., Two Stages Deep Learnign Approach 
        # of Carotid Intima-Media Thickness from Ultrasound Images
        '''
        self.translation = True
        self.h_flip = None
        self.v_flip = None
        self.rotation = None 
        '''

        if self.transform:
            img_roi = self.transform(img_roi)   # pillow -> tensor & 0 ~ 255  ->   -1 ~ 1
            img_li  = self.transform(img_li)
            img_ma  = self.transform(img_ma)
            #img_li.squeeze_()
            #img_ma.squeeze_()
        
        # unified
        #img_li_ma = img_li + 2.5*img_ma
        #img_li_ma = img_li_ma.to(dtype=torch.long)
        #return (img_roi, img_li_ma)

        # seperate w/ BCEWithLogitsLoss: this is numerically more stable
        img_li = img_li.to(dtype=torch.bool).to(dtype=torch.float)
        img_ma = img_ma.to(dtype=torch.bool).to(dtype=torch.float)

        '''
        np_li = (img_li*255).squeeze_()
        h, w = np_li.shape
        np_li = np_li.view(1,h,w).clamp_(0,255).numpy().astype(np.uint8)
        np_li = np_li.transpose(1,2,0)
        cv2.imwrite('1np_li.png', np_li)

        np_ma = (img_ma*255).squeeze_()
        h, w = np_ma.shape
        np_ma = np_ma.view(1,h,w).clamp_(0,255).numpy().astype(np.uint8)
        np_ma = np_ma.transpose(1,2,0)
        cv2.imwrite('1np_ma.png', np_ma)
        '''
        
        return (img_roi, img_li, img_ma)

        

def tensor2numpy(img):
    img = ((img*0.5) + 0.5).clamp(0.0, 1.0) # -1~1 -> 0 ~ 1
    # 0 ~ 1 -> 0 ~ 255  
    np_img = (img.cpu().detach() * 255.).numpy().astype(np.uint8)
    # C x H x W -> H x W x C
    np_img = np_img.transpose(1,2,0)[:,:,::-1]
    return np_img

if __name__ == '__main__':
    
    carotid_transform = transforms.Compose(
        [
            transforms.ToTensor(), # 0~255 -> 0~1
            #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) # 0~1 -> -1~1
        ])
    
    ROOT_IMAGE_DIR = './carotid_pp'
    JSON_PATH = 'gTruth_pp_small.json'
    train_set = CarotidSet(ROOT_IMAGE_DIR, JSON_PATH, transform=carotid_transform, 
                            flip=True, rotation=False, translation=False)

    for idx, data in enumerate(train_set):
        img, gt_li, gt_ma = data

        gt_li *= 255
        gt_ma *= 255
        img *= 255

        np_img = img.clamp_(0,255).numpy().astype(np.uint8)
        np_img = np_img.transpose(1,2,0) # N C H W -> H W C
        cv2.imwrite('test_img2.png', np_img)

        exit(1)
       
