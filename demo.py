import os, sys
import argparse
import cv2
import numpy as np
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from unet_model import UNet       # model
from dataset import CarotidSet    # custimized dataset
from utils import calc_iou, visualize_li_ma, visualize_gt_li_ma

from tqdm import tqdm

def opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='.carotid_pp')
    parser.add_argument('--json_test', type=str, default='./gTruth_pp_test.json')
    parser.add_argument('--log_dir', type=str, default='./vis')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--thres', type=float, default=0.5)
    return parser.parse_args()

def main():
    args = opt()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('can not use cuda')
        exit()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    transform = transforms.Compose([transforms.ToTensor()])
    test_set = CarotidSet(args.root, args.json_test, transform=transform,
                            flip=False, rotation=False, translation=False)
    
    model = UNet(1,2) # input 1ch # of class 2
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model'])
    else:
        print('please provise proper checkpoint path')
        exit(1)

    model = model.to(device)
    model = model.eval() 


    pbar = tqdm(total=len(test_set))
    for idx, data in enumerate(test_set):
        sample, gt_li, gt_ma = data  
        
        # 1 x H x W  -> N x 1 x H x W
        sample.unsqueeze_(dim=0)
        gt_li.unsqueeze_(dim=0)
        gt_ma.unsqueeze_(dim=0)

        sample = sample.to(device)
        gt_li = gt_li.to(device)   
        gt_ma = gt_ma.to(device)

        with torch.set_grad_enabled(False):
            logit = model(sample)

            pred_li = logit[:,0:1,:,:]
            pred_ma = logit[:,1:2,:,:]

        pred_li = pred_li.cpu()
        pred_ma = pred_ma.cpu()
        sample = sample.cpu()
        gt_li = gt_li.cpu()
        gt_ma = gt_ma.cpu()

        # image save
        np_pred_img = visualize_li_ma(pred_li, pred_ma, sample, args.thres)
        np_gt_img = visualize_gt_li_ma(gt_li, gt_ma, sample)

        final_img = cv2.hconcat([np_gt_img, np_pred_img])

        
        cv2.putText(final_img, f'thres:{args.thres}', (10,30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0,255,255), 2, cv2.LINE_AA)

        str_result_file_path = os.path.join(args.log_dir, str(idx+1)+'.png')
        cv2.imwrite(str_result_file_path, final_img)

        pbar.update()

if __name__ == '__main__':
    main()
