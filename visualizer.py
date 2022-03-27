import torch
import numpy as np
import cv2 
import json

from collections import deque

def parse_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    return data

def parse_attribute_from_json(json_dict, attribute):
    
    try:
        for line in json_dict['shapes']:
            if line['label'] == attribute:
                ret = line['points']
    except:
        return None

    return ret

def crop_img_using_pillow(img, roi):
    x = int(roi[0][0])
    y = int(roi[0][1])
    x2 = int(roi[1][0])# - x 
    y2 = int(roi[1][1])# - y
    roi = (x, y, x2, y2)
    
    cropped_img = img.crop(roi)
    return cropped_img




def _get_line_coords(img, post_proc=True):
    if not isinstance(img, np.ndarray):
        raise ValueError('input img must be numpy arrays, not {}'.format(type(img)))
    '''
        input
            img: numpy.ndarray (H x W x 1)
        output
            coords: list[tuple]
    '''
    height, width, _ = img.shape
    coords = []
    coords_dict = {} # x: y
    for h in range(height):
        for w in range(width):
            if img[h][w][0] == 255:
                coords.append((w,h))
                coords_dict[w] = h

    if not coords: # black image (no line)
        return []

    last_pt  = max(coords, key=lambda x: x[0])
    first_pt = min(coords, key=lambda x: x[0])

    connected_coords = []
    if post_proc:
        for x in range(first_pt[0], last_pt[0]+1):
            if not x in coords_dict.keys():
                coords_dict[x] = coords_dict[x-1]
        
    for x in coords_dict:
        connected_coords.append((x,coords_dict[x]))

    return connected_coords

def _draw_lima_line(src, coords, color):
    '''
        input
            src: numpy.ndarray (H x W x 3, unit8)
            coords: list[tuple(int, int)]
            color: tuple(int, int, int) # r g b
        output
            src: numpy.ndarray (H x W x 3, unit8)
    '''
    for x, y in coords:
        src[y][x][0] = color[0]
        src[y][x][1] = color[1]
        src[y][x][2] = color[2]
    return src   

def _paint_carotid_intima_media_region(src, coords_li, coords_ma, blend_weight, color=(90, 90, 255)):
    '''
        input
            src: numpy.ndarray (H x W x 3, uint8)
            coords_li: list[tuple(int, int)]
            coords_ma: list[tuple(int, int)]
        output
            dst: numpy.ndarray (H x W x 3, uint8)
    '''
    dst = src.copy()

    ma_coords_dict = {}
    for x, y in coords_ma:
        ma_coords_dict[x] = y

    avg_thickness = 0
    cnt = 0 
    for x, y in coords_li:
        if x in ma_coords_dict.keys():
            avg_thickness += (ma_coords_dict[x] - y)
            cnt += 1
    
    if cnt != 0:
        avg_thickness = avg_thickness//cnt
    
    for x, y in coords_li:
        if x in ma_coords_dict.keys():
            height = ma_coords_dict[x] - y
        else:
            height = avg_thickness

        for k in range(y, y+height+1):
            src[k][x][0] = color[0]
            src[k][x][1] = color[1]
            src[k][x][2] = color[2]

    dst = cv2.addWeighted(dst, blend_weight, src, 1.-blend_weight, 0)

    return dst

def visualize_gt_li_ma(gt_li, gt_ma, carotid_img, draw_line=True, blend_weight=0.5):
    n, c, h, w = carotid_img.shape # 1 1 H W (N == 1)

    carotid_img = _unnorm(carotid_img).view(1, h, w)
    carotid_img = torch.cat([carotid_img, carotid_img, carotid_img], dim=0) # 3 H W

    gt_li = (gt_li.view(1,h,w) * 255).to(carotid_img.dtype)
    gt_ma = (gt_ma.view(1,h,w) * 255).to(carotid_img.dtype)

    # tensor to numpy 
    np_carotid_img = (carotid_img.clamp_(0,255).numpy().astype(np.uint8)).transpose(1,2,0)
    np_gt_li = (gt_li.clamp_(0,255).numpy().astype(np.uint8)).transpose(1,2,0)
    np_gt_ma = (gt_ma.clamp_(0,255).numpy().astype(np.uint8)).transpose(1,2,0)

    # get line coords
    coords_li = _get_line_coords(np_gt_li)
    coords_ma = _get_line_coords(np_gt_ma)

    # paint carotid intima media region
    np_carotid_img = _paint_carotid_intima_media_region(np_carotid_img, coords_li, coords_ma, blend_weight)

    # draw line
    if draw_line:
        np_carotid_img = _draw_lima_line(np_carotid_img, coords_li, (255,0,0))
        np_carotid_img = _draw_lima_line(np_carotid_img, coords_ma, (0,0,255))
    
    return np_carotid_img




class Visualizer:
    
    def _to_bianry_mask(pred, thres):
        pred = torch.nn.functional.threshold(pred, thres, 0.)
        return pred.to(torch.bool)
    
    @staticmethod
    def visualize_pred(sample, color_dict, transparent,far_li=None, far_ma=None, near_li=None, near_ma=None, thres=0.2, draw_line=True):
        '''
            far, near li & ma : probablities
        '''
        blend_weight = transparent
        coords_dict = {'far':{}, 'near':{}}
        # tensor to numpy 
        if far_li is not None:
            far_li = (Visualizer._to_bianry_mask(far_li, thres) * 255).clamp_(0,255).numpy().astype(np.uint8).transpose(1,2,0)
            
            coords_dict['far']['li'] = _get_line_coords(far_li)

        if far_ma is not None:
            far_ma = (Visualizer._to_bianry_mask(far_ma, thres) * 255).clamp_(0,255).numpy().astype(np.uint8).transpose(1,2,0)
            coords_dict['far']['ma'] = _get_line_coords(far_ma)

        if near_li is not None:
            near_li = (Visualizer._to_bianry_mask(near_li, thres) * 255).clamp_(0,255).numpy().astype(np.uint8).transpose(1,2,0)
            coords_dict['near']['li'] = _get_line_coords(near_li)

        if near_ma is not None:
            near_ma = (Visualizer._to_bianry_mask(near_ma, thres) * 255).clamp_(0,255).numpy().astype(np.uint8).transpose(1,2,0)
            coords_dict['near']['ma'] = _get_line_coords(near_ma)

        # tensor to numpy 
        sample = torch.cat([sample, sample, sample], dim=0)
        sample = (sample*255).clamp_(0,255).numpy().astype(np.uint8).transpose(1,2,0)
        
        # draw IMT
        for m in ['far', 'near']:
            #for k in coords_dict[m]:
            sample = _paint_carotid_intima_media_region(
                src = sample, 
                coords_li = coords_dict[m]['li'], 
                coords_ma = coords_dict[m]['ma'], 
                blend_weight = blend_weight,
                color = color_dict[m]['imt']
            )

        # draw border line
        if draw_line:
            for m in ['far', 'near']:
                for k in coords_dict[m]:
                    color = color_dict[m][k]
                    sample = _draw_lima_line(sample, coords_dict[m][k], color)
            
        return sample

   
def bfs(seed_pt, mask, image, color, transparent):
    que = deque()

    height, width, _ = image.shape
    visited = [[False] * width for _ in range(height)]

    x, y = seed_pt
    que.append( (x,y) )
    visited[y][x] = True

    directions = [
        (-1,0),
        (1,0),
        (0,1),
        (0,-1)
    ]

    while que:
        x, y = que.pop()

        bgd_transparent = abs(1.0-transparent)

        if color[0] != 0:
            image[y][x][0] = int( image[y][x][0] * transparent + color[0] * bgd_transparent )

        if color[1] != 0:
            image[y][x][1] = int( image[y][x][1] * transparent + color[1] * bgd_transparent )
        
        if color[2] != 0:
            image[y][x][2] = int( image[y][x][2] * transparent + color[2] * bgd_transparent )

        for dx, dy in directions:
            nx = x + dx
            ny = y + dy

            if nx < 0 or ny < 0 or nx >= width or ny >=height:
                continue

            if visited[ny][nx] or mask[ny][nx][0] != 0:
                continue

            que.appendleft( (nx, ny) )
            visited[ny][nx] = True

    return image

def fill_gt_imt(image, lumen_intima, media_adventitia, color, transparent):    
    lumen_intima.sort()
    media_adventitia.sort()

    height, width, channel = image.shape # 3-channel
    mask = np.zeros(image.shape)

    
    # draw lines 
    for i in range(1, len(lumen_intima)):
        pt1 = lumen_intima[i-1]
        pt2 = lumen_intima[i]
        mask = cv2.line(mask, tuple(map(int, pt1)), tuple(map(int, pt2)), (255,255,255))

    
    for i in range(1, len(media_adventitia)):
        pt1 = media_adventitia[i-1]
        pt2 = media_adventitia[i]
        mask = cv2.line(mask, tuple(map(int, pt1)), tuple(map(int, pt2)), (255,255,255))
     
    # first pt
    mask = cv2.line(mask, tuple(map(int, lumen_intima[0])), tuple(map(int, media_adventitia[0])), (255,255,255))
    # last pt
    mask = cv2.line(mask, tuple(map(int, lumen_intima[-1])), tuple(map(int, media_adventitia[-1])), (255,255,255))

    # center location
    seed_pt = list(map(int, media_adventitia[len(media_adventitia)//2]))
    seed_pt[1] -= 2

    image = bfs(seed_pt, mask, image, color, transparent)
    
    return image


