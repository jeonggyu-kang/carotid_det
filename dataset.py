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
        self.resize = transforms.Resize(size=(128,128))
        #self.resize = transforms.Resize(size=(128,128))
        
    def __len__(self):
        return len(self.image_list)
        

    def make_gt(self, img_size, pt_list_x, pt_list_y):
        gt  = Image.new( mode = "L", size = img_size )
        # draw lines (LI & MA)


        if len(pt_list_x) != 0 and len(pt_list_y) != 0:
            

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




        fli_x = self.anno[index]['FLI']['x']
        fli_y = self.anno[index]['FLI']['y']
        fma_x = self.anno[index]['FMA']['x']
        fma_y = self.anno[index]['FMA']['y']


        nli_x = self.anno[index]['NLI']['x']
        nli_y = self.anno[index]['NLI']['y']
        nma_x = self.anno[index]['NMA']['x']
        nma_y = self.anno[index]['NMA']['y']


        
        # make LI & MA gt

        gt_images = []        
        gt_images.append(self.make_gt(img.size, nli_x, nli_y))
        gt_images.append(self.make_gt(img.size, nma_x, nma_y))
        gt_images.append(self.make_gt(img.size, fli_x, fli_y))
        gt_images.append(self.make_gt(img.size, fma_x, fma_y))

        # [1] translation
        if self.translation:   #! dynamic range adjustment
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
            
            for i in range(len(gt_images)):
                gt_images[i] = transforms.functional.affine(gt_images[i], degree, [0,0], 1.0, 0.0)

 



        img_roi = img.crop((roi[0], roi[1], roi[0] + roi[2], roi[1]+roi[3]))
        for i in range(len(gt_images)):
            gt_images[i] = gt_images[i].crop((roi[0], roi[1], roi[0] + roi[2], roi[1]+roi[3]))
 
        img_roi = self.resize(img_roi)
        for i in range(len(gt_images)):
            gt_images[i] = self.resize(gt_images[i])

        if self.h_flip:
            img_roi = self.h_flip(img_roi)
            for i in range(len(gt_images)):
                gt_images[i] = self.h_flip(gt_images[i])
        
        threshold = 50
        for i in range(len(gt_images)):
            gt_images[i] = gt_images[i].point(lambda p: p > threshold and 255)



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
            for i in range(len(gt_images)):
                gt_images[i] = self.transform(gt_images[i])
            #img_li.squeeze_()
            #img_ma.squeeze_()
        
        # unified
        #img_li_ma = img_li + 2.5*img_ma
        #img_li_ma = img_li_ma.to(dtype=torch.long)
        #return (img_roi, img_li_ma)

        # seperate w/ BCEWithLogitsLoss: this is numerically more stable
        for i in range(len(gt_images)):
            gt_images[i] = gt_images[i].to(dtype=torch.bool).to(dtype=torch.float)

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
        
        return (img_roi, gt_images[0], gt_images[1], gt_images[2], gt_images[3])

        

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
    JSON_PATH = 'gTruth_pp_v_test.json'
    train_set = CarotidSet(ROOT_IMAGE_DIR, JSON_PATH, transform=carotid_transform, 
                            flip=True, rotation=False, translation=False)

    for idx, data in enumerate(train_set):
        original_img, a, b, c, d = data

        annotated_imgs = [a,b,c,d]
        names = ['FLI','FMA','NLI','NMA']

        for name, img in zip(names, annotated_imgs):
            np_img = (img*255.0).clamp_(0,255).numpy().astype(np.uint8)
            np_img = np_img.transpose(1,2,0) # N C H W -> H W C
            file_name = name + str(idx+1) + '.png'
            cv2.imwrite(file_name, np_img)

       
