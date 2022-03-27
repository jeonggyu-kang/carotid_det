import os, sys
import argparse
import cv2
import numpy as np
import json

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from PIL import Image

from unet_model import UNet       # model
from dataset import CarotidSet    # custimized dataset
from visualizer import Visualizer, parse_json, parse_attribute_from_json, crop_img_using_pillow, fill_gt_imt

from tqdm import tqdm

def opt():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--src_json', type=str, default='./58.json') #! 환자 label me annotation 결과 json 파일
    parser.add_argument('--src_img', type=str, default='./58.bmp') #! (원본) 입력 경동맥 영상
    
    parser.add_argument('--dst_dir', type=str, default='./inference_result2')                    #! 결과 파일 저장 위치
    parser.add_argument('--pretrained', type=str, default='./work_dir/2022.02.10/ckpt/best_iou.pth')  #! 학습 결과 위치 
    parser.add_argument('--thres', type=float, default=0.2)                             #! 낮을수록 선이 두꺼움
    parser.add_argument('--num_classes', type=int, default=5)                           #! 클래스 개수

    return parser.parse_args()


def main():
    args = opt()
    transparent = 0.7
    roi_color = (0,0,255)   # ROI margin color
    args.color = { #! color dict
        'near' : {
            'li' : (255,0,0),
            'ma' : (128,0,0),
            'imt' : (0,0,255),
        },
        'far' : {
            'li' : (0,0,255),
            'ma' : (0,0,128),
            'imt' : (0,0,255),
        }
    }


    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        print('can not use cuda')
        exit()
    if not os.path.exists(args.dst_dir):
        os.makedirs(args.dst_dir)

    # json parsing
    patient_dict = parse_json(args.src_json)
    # roi 
    roi = parse_attribute_from_json(patient_dict, 'ROI')
    near_far_indicator = None
    # li
    for m, line_type in zip(['far', 'near'], ['FLI', 'NLI']):
        tmp = parse_attribute_from_json(patient_dict, line_type)
        if tmp is not None:
            near_far_indicator = m
            lumen_intima = tmp
            break
    # ma
    for m, line_type in zip(['far', 'near'], ['FMA', 'NMA']):
        tmp = parse_attribute_from_json(patient_dict, line_type)
        if tmp is not None:
            media_adventitia = tmp
            break

    # draw IMT (GT)
    # draw 
    gt_img = cv2.imread(args.src_img)
    '''
    for pt in lumen_intima:
        cv2.circle(gt_img, tuple(map(int, pt)), 3, (255,0,0), cv2.FILLED)
    for pt in media_adventitia:
        cv2.circle(gt_img, tuple(map(int, pt)), 3, (255,255,0), cv2.FILLED)
    '''
    right_image = fill_gt_imt(gt_img, lumen_intima, media_adventitia, args.color[near_far_indicator]['imt'], transparent)

    cropped_gt = crop_img_using_pillow(Image.fromarray(right_image), roi)
    cropped_gt = np.array(cropped_gt)


    # inference 
    transform = transforms.Compose([transforms.ToTensor()])
    
    model = UNet(1, args.num_classes)
    if args.pretrained:
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint['model'])
    else:
        print('please provise proper checkpoint path')
        exit(1)

    model = model.to(device)
    model = model.eval()
    
    # RoI cropping 
    full_image = Image.open(args.src_img)
    roi_cropped = crop_img_using_pillow(full_image, roi)
    

    sample = transform(roi_cropped)

    # 1 x H x W  -> N x 1 x H x W
    sample.unsqueeze_(dim=0)
    sample = sample.to(device)

    with torch.set_grad_enabled(False):
        logit = model(sample)

        logit = torch.sigmoid(logit)
        pred = logit.cpu()
        sample = sample.cpu()

        sample.squeeze_(dim=0)
        pred.squeeze_(dim=0)
        
    far_li = pred[0:1, :, :]
    far_ma = pred[1:2, :, :]
    near_li = pred[2:3, :, :]
    near_ma = pred[3:4, :, :]

    reseult_image = Visualizer.visualize_pred(
        sample=sample, 
        color_dict = args.color,
        transparent = transparent,
        far_li=far_li, far_ma=far_ma, 
        near_li=near_li, near_ma=near_ma, 
        thres=args.thres,
        draw_line = False
    )

    annotaed_image = Visualizer.visualize_pred(
        sample=sample, 
        color_dict = args.color,
        transparent = transparent,
        far_li=far_li, far_ma=far_ma, 
        near_li=near_li, near_ma=near_ma, 
        thres=args.thres,
        draw_line = False
    )

    # left image
    pt1, pt2 = roi
    left_image = cv2.imread(args.src_img)
    left_image = cv2.rectangle(left_image, tuple(map(int, pt1)), tuple(map(int, pt2)), roi_color, 2)

    # right image
    
    candidate_images = cv2.hconcat([ reseult_image, cropped_gt ])
    height_, width_, _ = candidate_images.shape
    candidate_images = cv2.resize(candidate_images, (width_*3, height_*3))

    while True:
                
        
        cv2.imshow('[I]: inference, [A]: annotated', candidate_images)

        user_key = cv2.waitKey(0) & 0xff

        if user_key == ord('i'): # inference
            right_image = reseult_image
            break
        elif user_key == ord('a'): # annotated 
            right_image = cropped_gt
            break
    
    

    # image write
    left_image_path  = os.path.join(args.dst_dir, 'left_image.bmp')
    i_right_image_path = os.path.join(args.dst_dir, 'right_image_inference.bmp')
    a_right_image_path = os.path.join(args.dst_dir, 'right_image_annotated.bmp')
    right_image_path = os.path.join(args.dst_dir, 'right_image.bmp')
    cv2.imwrite(left_image_path, left_image)
    cv2.imwrite(i_right_image_path, reseult_image)
    cv2.imwrite(a_right_image_path, cropped_gt)
    cv2.imwrite(right_image_path, right_image)

    print('Inference result has been written in {}'.format(args.dst_dir))
        
if __name__ == '__main__':
    main()
