import torch
import numpy as np
import cv2 

# (deprecated) domain specific evaluation metric
def calc_precision(pred, gt, thres=0.5) -> float: 
    #print(pred.shape) # torch.Size([4, 1, 128, 384]) 
    probs = torch.sigmoid(pred)
    probs = torch.nn.functional.threshold(probs, thres, 0.)
    probs = probs.to(torch.bool)

    gt = gt.to(torch.bool)

    hit_cnt = torch.sum(torch.logical_and(probs, gt))
    tot_cnt = torch.sum(gt)

    return hit_cnt.item() / tot_cnt.item()


def _logit2mask(logit, thres=0.5):
    probs = torch.sigmoid(logit)
    probs = torch.nn.functional.threshold(probs, thres, 0.)
    probs = probs.to(torch.bool)
    return probs

# This metric can sometimes provide misleading results
# when the class representation is small within the image,
# as the measure will be biased in mainly reporting 
# how well you identify negative case 
# (e.g. most cases are negative(bgd) in the predicted and ground-truth LI & MA images)
def calc_acc(logit, gt, thres=0.5) -> float:
    probs = torch.sigmoid(logit)
    probs = torch.nn.functional.threshold(probs, thres, 0.)
    probs = probs.to(torch.bool)
    
    gt = gt.to(torch.bool)

    n, c, h, w = logit.shape
    
    not_pred = torch.logical_not(probs.view(n,h,w))
    not_gt   = torch.logical_not(gt.view(n,h,w))

    tp = torch.logical_and(probs.view(n,h,w), gt.view(n,h,w))
    tn = torch.logical_and(not_pred, not_gt)
    fp = torch.logical_and(probs.view(n,h,w), not_gt)
    fn = torch.logical_and(not_pred, gt.view(n,h,w))
    
    # tp + tn / tp + tn + fp + fn
    tp_sum = (torch.sum(torch.sum(tp, dim=1), dim=1)).to(torch.float)
    tn_sum = (torch.sum(torch.sum(tn, dim=1), dim=1)).to(torch.float)

    fp_sum = (torch.sum(torch.sum(fp, dim=1), dim=1)).to(torch.float)
    fn_sum = (torch.sum(torch.sum(fn, dim=1), dim=1)).to(torch.float)    
    
    acc_score = torch.div( tp_sum+tn_sum, tp_sum+tn_sum+fp_sum+fn_sum )

    return torch.mean(acc_score)        

def calc_iou(pred, gt, thres=0.5) -> float:
    probs = torch.sigmoid(pred)
    probs = torch.nn.functional.threshold(probs, thres, 0.)
    probs = probs.to(torch.bool)

    gt = gt.to(torch.bool) #[N, 1, 128, 384]

    #print(probs)
    #print(gt)

    n, c, h, w = pred.shape

    intersection = torch.logical_and(pred.view(n,h,w), gt.view(n,h,w))
    union        = torch.logical_or(pred.view(n,h,w), gt.view(n,h,w))
            
    intersection_sum = (torch.sum(torch.sum(intersection, dim=1), dim=1)).to(torch.float)
    union_sum        = torch.sum( torch.sum(union,         dim=1), dim=1).to(torch.float)    
    iou_score        = torch.div(intersection_sum, union_sum)
    
    return torch.mean(iou_score) 
    

    
    
def _unnorm(tensor_img):
    return (tensor_img * 255).clamp_(0,255)


def _get_line_coords(img):
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

    last_pt  = max(coords, key=lambda x: x[0])
    first_pt = min(coords, key=lambda x: x[0])

    connected_coords = []
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

def visualize_li_ma(li_logits, ma_logits, carotid_img, thres, draw_line=True, blend_weight=0.5):
    n, c, h, w = carotid_img.shape

    li_mask = _logit2mask(li_logits, thres=thres).view(1,h,w)
    ma_mask = _logit2mask(ma_logits, thres=thres).view(1,h,w)

    carotid_img = _unnorm(carotid_img).view(1, h, w)
    carotid_img = torch.cat([carotid_img, carotid_img, carotid_img], dim=0) 
    
    # tensor to numpy
    np_carotid_img = (carotid_img.clamp_(0,255).numpy().astype(np.uint8)).transpose(1,2,0)
    np_pred_li = ((li_mask*255).clamp_(0,255).numpy().astype(np.uint8)).transpose(1,2,0)
    np_pred_ma = ((ma_mask*255).clamp_(0,255).numpy().astype(np.uint8)).transpose(1,2,0)

    # get line coords
    coords_li = _get_line_coords(np_pred_li)
    coords_ma = _get_line_coords(np_pred_ma)
    coords_dict = {}
    coords_dict['li'] = coords_li
    coords_dict['ma'] = coords_ma

    # paint carotid intima media region
    np_carotid_img = _paint_carotid_intima_media_region(np_carotid_img, coords_li, coords_ma, blend_weight)

    # draw line
    if draw_line:
        np_carotid_img = _draw_lima_line(np_carotid_img, coords_li, (0,255,0))
        np_carotid_img = _draw_lima_line(np_carotid_img, coords_ma, (0,0,255))

    return np_carotid_img, coords_dict


if __name__ == '__main__':
    pred = torch.ones((1,1,2,4), dtype=torch.float32)
    # batch size == 1
    pred[0][0][0][0] = 0
    pred[0][0][1][0] = 0
    pred[0][0][1][1] = 0
    
    gt = torch.ones((1,1,2,4), dtype=torch.long)
    # batch size == 1
    gt[0][0][0][0] = 0
    gt[0][0][0][1] = 0
    gt[0][0][1][0] = 0
    

    acc_score = calc_acc(pred, gt)
    print(acc_score)

    #iou_score = calc_iou(pred, gt)
    #print(iou_score)

    




