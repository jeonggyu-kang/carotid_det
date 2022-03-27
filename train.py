import os, sys 
import argparse 
import cv2      
import numpy as np   
import json

import copy
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

from tensorboardX import SummaryWriter  # tensorboard 

from unet_model import UNet     # model 
from dataset import CarotidSet  # customized dataset
from utils import calc_iou, calc_acc

def opt(): # import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='./hardsample_dataset_v5', help='path to dataset')
    parser.add_argument('--json_train', type=str, default='./gTruth_pp_v4.json', help='path to train dataset')
    parser.add_argument('--json_test', type=str, default='./gTruth_pp_test.json', help='path to train dataset')

    parser.add_argument('--max_epoch', type=int, default=50, help='max train epoch')

    parser.add_argument('--initial_lr', type=float, default=0.0001, help='initial learning rate')

    parser.add_argument('--batch_size', type=int, default=16, help='batch size')

    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers')
    parser.add_argument('--log_dir', type=str, default='./work_dir/2022.02.10', help='path to log directory')

    parser.add_argument('--resume_path', type=str, default=None, help='path to resume model')
    return parser.parse_args()

###################################################################
#* model train function (single epoch)
###################################################################
def train(device, model, optimizer, criterion, scheduler, data_loader, data_set, writer, epoch):
    model = model.to(device)
    model.train()

    running_loss = 0.0

    far_running_iou_li = 0.0
    far_running_iou_ma = 0.0

    near_running_iou_li = 0.0
    near_running_iou_ma = 0.0

    far_running_acc_li = 0.0
    far_running_acc_ma = 0.0
    near_running_acc_li = 0.0
    near_running_acc_ma = 0.0

    #* [5-2] singe epoch (train or validation)
    global_step = 0
    for idx, data in enumerate(data_loader, 0):
        #* [5-2.1] input samples, labels # # raw-roi, FLI, FMA, NLI, NMA
        sample, gt_li_far, gt_ma_far, gt_li_near, gt_ma_near = data
                
        sample = sample.to(device)
        gt_li_far = gt_li_far.to(device)
        gt_ma_far = gt_ma_far.to(device)
        gt_li_near = gt_li_near.to(device)
        gt_ma_near = gt_ma_near.to(device)
                        
        #* [5-2.2] forward pass and backward pass, loss, and gradient update
        # set gradient to zero
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward pass 
            logits = model(sample)
            #probs  = torch.nn.functional.sigmoid(logits)

            # calcuate loss
            far_logits_li = logits[:, 0:1, :, :]
            far_logits_ma = logits[:, 1:2, :, :]
            near_logits_li = logits[:, 2:3, :, :]
            near_logits_ma = logits[:, 3:4, :, :]

            loss  = criterion(far_logits_li, gt_li_far)
            loss += criterion(far_logits_ma, gt_ma_far)
            loss += criterion(near_logits_li, gt_li_near)
            loss += criterion(near_logits_ma, gt_ma_near)

            # calculate iou
            far_running_iou_li += calc_iou(logits[:, 0:1, :, :].detach(), gt_li_far.detach(), thres=0.5)
            far_running_iou_ma += calc_iou(logits[:, 1:2, :, :].detach(), gt_ma_far.detach(), thres=0.5)
            near_running_iou_li += calc_iou(logits[:, 2:3, :, :].detach(), gt_li_near.detach(), thres=0.5)
            near_running_iou_ma += calc_iou(logits[:, 3:4, :, :].detach(), gt_ma_near.detach(), thres=0.5)
            
            # calculate accuracy
            far_running_acc_li += calc_acc(logits[:, 0:1, :, :].detach(), gt_li_far.detach(), thres=0.5)
            far_running_acc_ma += calc_acc(logits[:, 1:2, :, :].detach(), gt_ma_far.detach(), thres=0.5)
            near_running_acc_li += calc_acc(logits[:, 2:3, :, :].detach(), gt_li_near.detach(), thres=0.5)
            near_running_acc_ma += calc_acc(logits[:, 3:4, :, :].detach(), gt_ma_near.detach(), thres=0.5)
            
            # backward pass
            loss.backward()

            # update gradient
            optimizer.step()

        #* [5-2.3] running loss and accuracy
        # statistics
        running_loss += loss.item()
        writer.add_scalar('loss/ce_loss', loss.item(), epoch*len(data_loader) + idx)

        # print loss every 1000 steps
        if (idx+1) % 50 == 0: 
            print("{} - [Epoch-{}][{}/{}] loss: {:.2f}".format(
                datetime.now(), epoch, idx+1, len(data_loader), running_loss))
            running_loss = 0.0

        global_step += 1

    #* [5-2-END] singe epoch (train or validation) END
    if scheduler is not None:
        scheduler.step()

    far_train_iou_li = far_running_iou_li.item()
    far_train_iou_ma = far_running_iou_ma.item()
    near_train_iou_li = near_running_iou_li.item()
    near_train_iou_ma = near_running_iou_ma.item()

    #* [5-3] print epoch accuracy
    far_train_iou_li = far_running_iou_li / global_step
    far_train_iou_ma = far_running_iou_ma / global_step
    near_train_iou_li = near_running_iou_li / global_step
    near_train_iou_ma = near_running_iou_ma / global_step

    far_train_acc_li = far_running_acc_li / global_step
    far_train_acc_ma = far_running_acc_ma / global_step
    near_train_acc_li = near_running_acc_li / global_step
    near_train_acc_ma = near_running_acc_ma / global_step

    print('[{}/{}]   Near:(IoU/Acc) {:.4f}/{:.4f} Far:(IoU/Acc) {:.4f}/{:.4f}  '.format(
        epoch, 50, 
        (near_train_iou_li+near_train_iou_ma)/2, (near_train_acc_li+near_train_acc_ma)/2,
        (far_train_iou_li+far_train_iou_ma)/2, (far_train_acc_li+far_train_acc_ma)/2
    ))

    writer.add_scalar('train/IoU/near/li', near_train_iou_li, epoch*len(data_loader))
    writer.add_scalar('train/IoU/near/ma', near_train_iou_ma, epoch*len(data_loader))
    writer.add_scalar('train/IoU/far/li', far_train_iou_li, epoch*len(data_loader))
    writer.add_scalar('train/IoU/far/ma', far_train_iou_ma, epoch*len(data_loader))

    writer.add_scalar('train/Acc/near/li', near_train_acc_li, epoch*len(data_loader))
    writer.add_scalar('train/Acc/near/ma', near_train_acc_ma, epoch*len(data_loader))
    writer.add_scalar('train/Acc/far/li', far_train_acc_li, epoch*len(data_loader))
    writer.add_scalar('train/Acc/far/ma', far_train_acc_ma, epoch*len(data_loader))


###################################################################
#* validation test function
###################################################################
def test(device, model, criterion, data_loader, data_set, writer, epoch):
    model = model.to(device)
    model.eval()
    
    running_loss = 0.0

    far_running_iou_li = 0.0
    far_running_iou_ma = 0.0

    near_running_iou_li = 0.0
    near_running_iou_ma = 0.0

    far_running_acc_li = 0.0
    far_running_acc_ma = 0.0
    near_running_acc_li = 0.0
    near_running_acc_ma = 0.0

    global_step = 0
    for idx, data in enumerate(data_loader, 0):
        #* [5-2.1] input samples, labels
        sample, gt_li_far, gt_ma_far, gt_li_near, gt_ma_near = data
                
        sample = sample.to(device)
        gt_li_far = gt_li_far.to(device)
        gt_ma_far = gt_ma_far.to(device)
        gt_li_near = gt_li_near.to(device)
        gt_ma_near = gt_ma_near.to(device)

        #* [5-2.2] forward pass and backward pass, loss, and gradient update
        with torch.set_grad_enabled(False):
            # forward pass 
            logits = model(sample)
            #probs  = torch.nn.functional.sigmoid(logits)   

            # calculate iou
            far_running_iou_li += calc_iou(logits[:, 0:1, :, :].detach(), gt_li_far.detach(), thres=0.5)
            far_running_iou_ma += calc_iou(logits[:, 1:2, :, :].detach(), gt_ma_far.detach(), thres=0.5)
            near_running_iou_li += calc_iou(logits[:, 2:3, :, :].detach(), gt_li_near.detach(), thres=0.5)
            near_running_iou_ma += calc_iou(logits[:, 3:4, :, :].detach(), gt_ma_near.detach(), thres=0.5)
            
            # calculate accuracy
            far_running_acc_li += calc_acc(logits[:, 0:1, :, :].detach(), gt_li_far.detach(), thres=0.5)
            far_running_acc_ma += calc_acc(logits[:, 1:2, :, :].detach(), gt_ma_far.detach(), thres=0.5)
            near_running_acc_li += calc_acc(logits[:, 2:3, :, :].detach(), gt_li_near.detach(), thres=0.5)
            near_running_acc_ma += calc_acc(logits[:, 3:4, :, :].detach(), gt_ma_near.detach(), thres=0.5)

        global_step += 1

    far_train_iou_li = far_running_iou_li.item()
    far_train_iou_ma = far_running_iou_ma.item()
    near_train_iou_li = near_running_iou_li.item()
    near_train_iou_ma = near_running_iou_ma.item()

    #* [5-3] print epoch accuracy
    far_train_iou_li = far_running_iou_li / global_step
    far_train_iou_ma = far_running_iou_ma / global_step
    near_train_iou_li = near_running_iou_li / global_step
    near_train_iou_ma = near_running_iou_ma / global_step

    far_train_acc_li = far_running_acc_li / global_step
    far_train_acc_ma = far_running_acc_ma / global_step
    near_train_acc_li = near_running_acc_li / global_step
    near_train_acc_ma = near_running_acc_ma / global_step

    print('Test [{}/{}]   Near:(IoU/Acc) {:.4f}/{:.4f} Far:(IoU/Acc) {:.4f}/{:.4f}  '.format(
        epoch, 50, 
        (near_train_iou_li+near_train_iou_ma)/2, (near_train_acc_li+near_train_acc_ma)/2,
        (far_train_iou_li+far_train_iou_ma)/2, (far_train_acc_li+far_train_acc_ma)/2
    ))

    writer.add_scalar('test/IoU/near/li', near_train_iou_li, epoch*len(data_loader))
    writer.add_scalar('test/IoU/near/ma', near_train_iou_ma, epoch*len(data_loader))
    writer.add_scalar('test/IoU/far/li', far_train_iou_li, epoch*len(data_loader))
    writer.add_scalar('test/IoU/far/ma', far_train_iou_ma, epoch*len(data_loader))

    writer.add_scalar('test/Acc/near/li', near_train_acc_li, epoch*len(data_loader))
    writer.add_scalar('test/Acc/near/ma', near_train_acc_ma, epoch*len(data_loader))
    writer.add_scalar('test/Acc/far/li', far_train_acc_li, epoch*len(data_loader))
    writer.add_scalar('test/Acc/far/ma', far_train_acc_ma, epoch*len(data_loader))

    avg_iou = (near_train_iou_li + near_train_iou_ma + far_train_iou_li + far_train_iou_ma) / 4
    avg_acc = (near_train_acc_li + near_train_acc_ma + far_train_acc_li + far_train_acc_ma) / 4

    return avg_iou, avg_acc


def main():
    args = opt()
    
    if torch.cuda.is_available(): # True False
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('can not use CUDA.')
        exit(1)

    # writer
    if not os.path.exists(os.path.join(args.log_dir, 'ckpt')):
        os.makedirs(os.path.join(args.log_dir, 'ckpt'))
    writer = SummaryWriter(args.log_dir)
    
    # datalaoder
    carotid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
        ])
    
    train_set = CarotidSet(args.root, args.json_train, transform=carotid_transform, flip=True, rotation=True, translation=False)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    test_set = CarotidSet(args.root, args.json_test, transform=carotid_transform, flip=False, rotation=False, translation=False)
    testloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=False, num_workers=1)
    
    data_loader   = {'train': trainloader  ,    'val': testloader}
    dataset_sizes = {'train': len(train_set),   'val': len(test_set)} #-> 10000
    dataset_steps = {'train': len(trainloader), 'val': len(testloader)} #-> 100
    print('dataset sample size: {}'.format(dataset_sizes))
    print('dataset step size: {}'.format(dataset_steps))
   
    ###################################################################
    #* [4] Network (or Model) + model load + optimizer + loss fucntion + learning rate scheduler
    ###################################################################
    # netweork 
    num_classes = 5 # pixel-wise
    model = UNet(1, num_classes)
    

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    #optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)

    # scheduler
    if args.max_epoch == 200:
        milestones = [ 50, 100, 150 ]
    elif args.max_epoch == 50:
        milestones = [ x for x in range(8, 50, 8)]
    elif args.max_epoch == 100: 
        milestones = [ x for x in range(8, 100, 8)]
    else:
        raise NotImplementedError
    
    #scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.5)
    scheduler = None
    #for epoch in range(1, 200):
    #    scheduler.step()
    #    print('Epoch {}, lr {:.5f}'.format(epoch, optimizer.param_groups[0]['lr']))
    #exit()

    #! this feature has not been implemented yet.
    if args.resume_path is not None:
        checkpoint = torch.load(args.resume_path) # load file
    
        model.load_state_dict(checkpoint['model'])          # load model
        optimizer.load_state_dict(checkpoint['optimizer'])   # load optimize
        scheduler.load_state_dict(checkpoint['scheduler'])  # load scheduler

        #optimizer.load_state_dict(checkpoint['optimizer'])
        model = model.to(device)
        #optimizer = optimizer.to(device) # 1.3.1 -> 1.8.0

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        train_logger.info("{} - loaded checkpoint from {}, start epoch: {}".format(
            datetime.now(), args.resume_path, checkpoint['epoch']))
        
        start_epoch = checkpoint['epoch']
        end_epoch = args.max_epoch
    else:
        start_epoch = 1
        end_epoch = args.max_epoch
    
    
    model = model.to(device) # DDP or single gpu?
    best_acc = 0.0
    best_iou = 0.0

    for epoch in range(start_epoch, end_epoch+1):
        # train for single epoch
        train(device, model, optimizer, criterion, scheduler, trainloader, train_set, writer, epoch)
        
        if epoch % 10 == 0:  # test every
            avg_iou, avg_acc = test(device, model, criterion, testloader, test_set, writer, epoch)
            # deep copy best model
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_acc_model = copy.deepcopy(model.state_dict())
            if avg_iou > best_iou:
                best_iou = avg_iou
                best_iou_model = copy.deepcopy(model.state_dict())


        # save model every 10 epoch
        if epoch % 10 == 0: 
            save_model_path = os.path.join(os.path.join(args.log_dir, 'ckpt'), str(epoch) + '.pth')
            torch.save({
                'model': model.state_dict()
            }, save_model_path)
            

    #* [5-END] train loop END
    
    # save best model iou
    save_model_path = os.path.join(os.path.join(args.log_dir, 'ckpt'), 'best_iou.pth')
    model.load_state_dict(best_iou_model)
    torch.save({
        'model': model.state_dict()
    }, save_model_path)
    # save best model acc
    save_model_path = os.path.join(os.path.join(args.log_dir, 'ckpt'), 'best_acc.pth')
    model.load_state_dict(best_acc_model)
    torch.save({
        'model': model.state_dict()
    }, save_model_path)
    


    print("Finished Training! - best IoU: {:.4f}   best Acc: {:.4f}".format(best_iou, best_acc))

    #train_log_path = os.path.join(args.log_dir, 'train_log.txt')
    #with open(train_log_path, 'w') as f:
    #    f.write('[Val] best acc: {}'.format(best_acc*100))
    writer.close()

if __name__ == '__main__':
    main()
