"""
from datetime import MAXYEAR
import torch
from torch import torch.nn as nn

import os, sys
import numpy as np
import cv2

from torch.utils.dta import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
"""

import json
def parse_json(json_file):
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    for idx in range(len(json_data['roi'])):
        json_data['roi'][idx] = list(map(int, json_data['roi'][idx]))
    
    return json_data['file_name'], json_data['roi']


if __name__ == '__main__':
    file_names, rois = parse_json('carotid.json')
    print(file_names)
    print(rois)
    


"""
class CarotidDataset(Dataset): #json, xml, hdf5, html, url # init json viersion (Git) + @ ver2
    def __init__(self, root, train, rotation, translation, flip):
        self.root = root # /downloads/annotation.mat
        self.train = train
        self.rotation = rotation
        self.translation = translation
        self.flip = flip # horizontal flip

        # TODO 
        file_path (cv2)

        self.file_path_list = [] #'./dataset/images/xxxx.jpg png
        self.rois = [] # [345, 278, 2234, 302]
        self.type = [] # 0 1 2 3 4
        self.LI = [] # {'x':[1,2,3,]},'y':[1,2,3,4,5]}
        slef.MA
        assert len(self.file_path_list) == len(self.rois)
        # 배지어 곡선


        roi, type     /   LI MA



        #def parse_mat_data(self.root):
                source -> file path -> 상대경로로 변경 예정
                label definitions -> MidCCA, DistalCCA, Blud, ICA -> color
                LI, MA
                label data (statge-1, result, type 정보 인가?)  LI, MA groud truth
                                               + type 정보 맞추게

        #    return

    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, index):   # index 번째 데이터 샘플과 그에 따른 정답을  return
        # augmentation 기법 적용!
        # shuffle
        if self.train:
            image = cv2.imaged(self.file_path_list[index])

            if self.rotation and ... > 0.5:
                image = do_rotation(image, self.rois[index],self.LI[index],self.MA[index])
            if self.translation and ... > 0.5:
                image = do_translation(image)
            if self.flip and ... > 0.5:
                image = do_flip(image)

            result = {
                      'img' : image,
                      'roi' : self.rois[index],
                      'l_type' : self.type[index],
                      'LI' : self.LI[index],
                      'MA' : self.MA[index]
            }


        else: # test
            image = cv2.imaged(self.file_path_list[index])
            # TODO 0 ~ 255 -> 0 ~1 or -1 ~1
            # image = whitening ??

            result = {
                      'img' : image,
                      'roi' : self.rois[index],
                      'l_type' : self.type[index],
                      'LI' : self.LI[index],
                      'MA' : self.MA[index]
            }

        return result




        test mod?
"""