import torch
import numpy as np

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


def _logit2mask(logit, thres=0.1):
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
    
def visualize_li_ma(li_logits, ma_logits, carotid_img):
    li_mask = _logit2mask(li_logits)
    ma_mask = _logit2mask(ma_logits)

    #binary -> 255 add 1ch
    n, c, h, w = carotid_img.shape

    carotid_img = _unnorm(carotid_img) # 0~1 -> 0~255
    
    # 1 x 1 x h x w -> 3 x h x w
    carotid_img = carotid_img.view(1, h, w)
    carotid_img = torch.cat([carotid_img, carotid_img, carotid_img], dim=0) 

    li_mask = li_mask.view(1,h,w)
    li_mask = (li_mask*255).to(carotid_img.dtype)
    ma_mask = ma_mask.view(1,h,w)
    ma_mask = (ma_mask*255).to(carotid_img.dtype)

    '''
    ma_mask = ma_mask.clamp_(0,255).numpy().astype(np.uint8)
    ma_mask = ma_mask.transpose(1,2,0)
    return ma_mask
    '''

    # mark lumen intia and media adventitia
    carotid_img[0:1, :, :] = li_mask # lumen intima - blue
    carotid_img[2:3, :, :] = ma_mask # media adventitia - red

    np_carotid_img = carotid_img.clamp_(0,255).numpy().astype(np.uint8)
    np_carotid_img = np_carotid_img.transpose(1,2,0)
    return np_carotid_img


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

    




