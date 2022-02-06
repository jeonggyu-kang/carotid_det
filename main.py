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

    parser.add_argument('--root', type=str, default='./hardsample_dataset', help='path to dataset')
    parser.add_argument('--json_train', type=str, default='./gTruth_pp_v3.json', help='path to train dataset')
    parser.add_argument('--json_test', type=str, default='./gTruth_pp_test.json', help='path to train dataset')

    parser.add_argument('--max_epoch', type=int, default=50, help='max train epoch')

    parser.add_argument('--initial_lr', type=float, default=0.0001, help='initial learning rate')

    parser.add_argument('--batch_size', type=int, default=8, help='batch size')

    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers')
    parser.add_argument('--log_dir', type=str, default='./work_dir', help='path to log directory')

    parser.add_argument('--resume_path', type=str, default=None, help='path to resume model')
    return parser.parse_args()

###################################################################
#* model train function (single epoch)
###################################################################
def train(device, model, optimizer, criterion, scheduler, data_loader, data_set, writer, epoch):
    model = model.to(device)
    model.train()

    running_loss = 0.0

    near_running_iou_li = 0.0
    near_running_iou_ma = 0.0

    near_running_acc_li = 0.0
    near_running_acc_ma = 0.0

    far_running_iou_li = 0.0
    far_running_iou_ma = 0.0

    far_running_acc_li = 0.0
    far_running_acc_ma = 0.0


    #* [5-2] singe epoch (train or validation)
    for idx, data in enumerate(data_loader, 0):
        #* [5-2.1] input samples, labels
        sample, near_gt_li, near_gt_ma, far_gt_li, far_gt_ma = data
                
        sample = sample.to(device)
        near_gt_li = near_gt_li.to(device)
        near_gt_ma = near_gt_ma.to(device)

        far_gt_li = far_gt_li.to(device)
        far_gt_ma = far_gt_ma.to(device)        
                        
        #* [5-2.2] forward pass and backward pass, loss, and gradient update
        # set gradient to zero
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward pass 
            logits = model(sample)
            #probs  = torch.nn.functional.sigmoid(logits)

            # calcuate loss
            near_logits_li = logits[:, 0:1, :, :]
            near_logits_ma = logits[:, 1:2, :, :]

            far_logits_li = logits[:, 2:3, :, :]
            far_logits_ma = logits[:, 3:4, :, :]

            loss  = criterion(near_logits_li, near_gt_li)
            loss += criterion(near_logits_ma, near_gt_ma)

            loss += criterion(far_logits_li, far_gt_li)
            loss += criterion(far_logits_ma, far_gt_ma)


            # calculate iou
            near_running_iou_li += calc_iou(logits[:, 0:1, :, :].detach(), near_gt_li.detach(), thres=0.5)
            near_running_iou_ma += calc_iou(logits[:, 1:2, :, :].detach(), near_gt_ma.detach(), thres=0.5)
            far_running_iou_li  += calc_iou(logits[:, 2:3, :, :].detach(), far_gt_li.detach(), thres=0.5)
            far_running_iou_ma  += calc_iou(logits[:, 3:4, :, :].detach(), far_gt_ma.detach(), thres=0.5)
            # calculate accuracy
            near_running_acc_li += calc_acc(logits[:, 0:1, :, :].detach(), near_gt_li.detach(), thres=0.5)
            near_running_acc_ma += calc_acc(logits[:, 1:2, :, :].detach(), near_gt_ma.detach(), thres=0.5)
            far_running_acc_li  += calc_acc(logits[:, 2:3, :, :].detach(), far_gt_li.detach(), thres=0.5)
            far_running_acc_ma  += calc_acc(logits[:, 3:4, :, :].detach(), far_gt_ma.detach(), thres=0.5)
            
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
    #* [5-2-END] singe epoch (train or validation) END
    scheduler.step()

    #* [5-3] print epoch accuracy
    near_epoch_iou_li = near_running_iou_li / len(data_loader)
    near_epoch_iou_ma = near_running_iou_ma / len(data_loader)
    far_epoch_iou_li  = far_running_iou_li / len(data_loader)
    far_epoch_iou_ma  = far_running_iou_ma / len(data_loader)

    near_epoch_acc_li = near_running_acc_li / len(data_loader)
    near_epoch_acc_ma = near_running_acc_ma / len(data_loader)
    far_epoch_acc_li  = far_running_acc_li / len(data_loader)
    far_epoch_acc_ma  = far_running_acc_ma / len(data_loader)

        
    print("{0} - [{1}-Epoch-{2}] (Near)IoU(LI-MA): {3:.2f}, {4:.2f} (Far)IoU(LI-MA): {5:.2f}, {6:.2f}, LR: {7:.5f}".format(
        datetime.now(), 'Train', epoch, 
        near_epoch_iou_li.item()*100.0, near_epoch_iou_ma.item()*100.0, 
        far_epoch_iou_li.item()*100.0, far_epoch_iou_ma.item()*100.0, 
        optimizer.param_groups[0]['lr']))
    print("{0} - [{1}-Epoch-{2}] (Near)Acc(LI-MA): {3:.2f}, {4:.2f} (Far)Acc(LI-MA): {5:.2f}, {6:.2f}, LR: {6:.5f}".format(
        datetime.now(), 'Train', epoch, 
        near_epoch_acc_li.item()*100.0, near_epoch_acc_ma.item()*100.0, 
        far_epoch_acc_li.item()*100.0, far_epoch_acc_ma.item()*100.0, 
        optimizer.param_groups[0]['lr']))

    writer.add_scalar('IoU/train/near/li', near_epoch_iou_li.item()*100.0, epoch*len(data_loader))
    writer.add_scalar('IoU/train/near/ma', near_epoch_iou_ma.item()*100.0, epoch*len(data_loader))
    writer.add_scalar('IoU/train/far/li', far_epoch_iou_li.item()*100.0, epoch*len(data_loader))
    writer.add_scalar('IoU/train/far/ma', far_epoch_iou_ma.item()*100.0, epoch*len(data_loader))

    writer.add_scalar('Acc/train/near/li', near_epoch_acc_li.item()*100.0, epoch*len(data_loader))
    writer.add_scalar('Acc/train/near/ma', near_epoch_acc_ma.item()*100.0, epoch*len(data_loader))
    writer.add_scalar('Acc/train/far/li', far_epoch_acc_li.item()*100.0, epoch*len(data_loader))
    writer.add_scalar('Acc/train/far/ma', far_epoch_acc_ma.item()*100.0, epoch*len(data_loader))


###################################################################
#* validation test function
###################################################################
def test(device, model, criterion, data_loader, data_set, writer, epoch):
    model = model.to(device)
    model.eval()
    
    running_iou_li = 0.0
    running_iou_ma = 0.0

    running_acc_li = 0.0
    running_acc_ma = 0.0

    #* [5-2] singe epoch (train or validation)
    for idx, data in enumerate(data_loader, 0):
        #* [5-2.1] input samples, labels
        sample, near_gt_li, near_gt_ma, far_gt_li, far_gt_ma = data
                
        sample = sample.to(device)
        near_gt_li = near_gt_li.to(device)
        near_gt_ma = near_gt_ma.to(device)

        far_gt_li = far_gt_li.to(device)
        far_gt_ma = far_gt_ma.to(device)   

        #* [5-2.2] forward pass and backward pass, loss, and gradient update
        with torch.set_grad_enabled(False):
            # forward pass 
            logits = model(sample)
            #probs  = torch.nn.functional.sigmoid(logits)   

            #* [5-2.3] running loss and accuracy
            near_running_iou_li += calc_iou(logits[:, 0:1, :, :], near_gt_li, thres=0.5)
            near_running_iou_ma += calc_iou(logits[:, 1:2, :, :], near_gt_ma, thres=0.5)
            far_running_iou_li  += calc_iou(logits[:, 2:3, :, :], far_gt_li, thres=0.5)
            far_running_iou_ma  += calc_iou(logits[:, 3:4, :, :], far_gt_ma, thres=0.5)
            # calculate accuracy
            near_running_acc_li += calc_acc(logits[:, 0:1, :, :], near_gt_li, thres=0.5)
            near_running_acc_ma += calc_acc(logits[:, 1:2, :, :], near_gt_ma, thres=0.5)
            far_running_acc_li  += calc_acc(logits[:, 2:3, :, :], far_gt_li, thres=0.5)
            far_running_acc_ma  += calc_acc(logits[:, 3:4, :, :], far_gt_ma, thres=0.5)
            
    
    #* [5-3] print epoch accuracy
    near_epoch_iou_li = near_running_iou_li / len(data_loader)
    near_epoch_iou_ma = near_running_iou_ma / len(data_loader)
    far_epoch_iou_li  = far_running_iou_li / len(data_loader)
    far_epoch_iou_ma  = far_running_iou_ma / len(data_loader)

    near_epoch_acc_li = nearrunning_acc_li / len(data_loader)
    near_epoch_acc_ma = nearrunning_acc_ma / len(data_loader)
    far_epoch_acc_li  = far_running_acc_li / len(data_loader)
    far_epoch_acc_ma  = far_running_acc_ma / len(data_loader)

    print("{0} - [{1}-Epoch-{2}] (Near)IoU(LI-MA): {3:.2f}, {4:.2f} (Far)IoU(LI-MA): {5:.2f}, {6:.2f}".format(
        datetime.now(), 'Validation', epoch, 
        near_epoch_iou_li.item()*100.0, near_epoch_iou_ma.item()*100.0,
        far_epoch_iou_li.item()*100.0, far_epoch_iou_ma.item()*100.0
    ))
    print("{0} - [{1}-Epoch-{2}] (Near)IoU(LI-MA): {3:.2f}, {4:.2f} (Far)IoU(LI-MA): {5:.2f}, {6:.2f}".format(
        datetime.now(), 'Validation', epoch,
        near_epoch_iou_ma.item()*100.0, near_epoch_iou_ma.item()*100.0,
        far_epoch_iou_ma.item()*100.0, far_epoch_iou_ma.item()*100.0
    ))
    print('='*80)    
    
    writer.add_scalar('IoU/val/near/li', near_epoch_iou_li.item() * 100.0, epoch*len(data_loader))
    writer.add_scalar('IoU/val/near/ma', near_epoch_iou_ma.item() * 100.0, epoch*len(data_loader))
    writer.add_scalar('IoU/val/far/li', far_epoch_iou_li.item() * 100.0, epoch*len(data_loader))
    writer.add_scalar('IoU/val/far/ma', far_epoch_iou_ma.item() * 100.0, epoch*len(data_loader))

    writer.add_scalar('Acc/val/near/li', near_epoch_acc_li.item() * 100.0, epoch*len(data_loader))
    writer.add_scalar('Acc/val/near/ma', near_epoch_acc_ma.item() * 100.0, epoch*len(data_loader))
    writer.add_scalar('Acc/val/far/li', far_epoch_acc_li.item() * 100.0, epoch*len(data_loader))
    writer.add_scalar('Acc/val/far/ma', far_epoch_acc_ma.item() * 100.0, epoch*len(data_loader))
    
    return (near_epoch_iou_li.item() + near_epoch_iou_ma.item() + far_epoch_iou_li.item() + far_epoch_iou_ma.item())/4


def main():
    ###################################################################
    #* [1] parse input arguments: batch size, shuffle mode, the number of workers..
    #*     & device check 
    ###################################################################
    args = opt()
    
    if torch.cuda.is_available(): # True False
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print('can not use CUDA.')
        exit(1)

    ###################################################################
    #* [2] logger + tensorboard(X)
    ###################################################################
    if not os.path.exists(os.path.join(args.log_dir, 'ckpt')):
        os.makedirs(os.path.join(args.log_dir, 'ckpt'))
    writer = SummaryWriter(args.log_dir)
    
    ###################################################################
    #* [3] data loader (transform, shuffle, data augmentation..)
    ###################################################################
    carotid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            #transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) 
        ])
    
    train_set = CarotidSet(args.root, args.json_train, transform=carotid_transform, flip=False, rotation=False, translation=False)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # TODO : generate seperate json file (test)
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
    num_classes = 4 # pixel-wise
    model = UNet(1, num_classes)
    #print(model)

    #criterion = nn.CrossEntropyLoss()
    criterion = nn.BCEWithLogitsLoss()

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.initial_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    # scheduler
    if args.max_epoch == 200:
        milestones = [ 50, 100, 150 ]
    elif args.max_epoch == 50:
        milestones = [ x for x in range(8, 50, 8)]
    elif args.max_epoch == 200: # sanity check
        print('performing sanity check')
        milestones = [ 80, 160 ]
    else:
        raise NotImplementedError
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=0.5)
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
    
    ###################################################################
    #* [5] train loop
    ###################################################################
    model = model.to(device) # DDP or single gpu?
    best_acc = 0.0
    for epoch in range(start_epoch, end_epoch+1):
        #* [5-1] train or validation

        # train for single epoch
        train(device, model, optimizer, criterion, scheduler, trainloader, train_set, writer, epoch)
        
        # test model accuracy every 20 epoch
        if epoch % 20 == 0: 
            validation_acc = test(device, model, criterion, testloader, test_set, writer, epoch)
            # deep copy best model
            if validation_acc > best_acc:
                best_acc = validation_acc
                best_model = copy.deepcopy(model.state_dict())

        # save model every 10 epoch
        if epoch % 20 == 0: 
            save_model_path = os.path.join(os.path.join(args.log_dir, 'ckpt'), str(epoch) + '.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            }, save_model_path)
            

    #* [5-END] train loop END
    
    # save best model
    save_model_path = os.path.join(os.path.join(args.log_dir, 'ckpt'), 'best.pth')
    model.load_state_dict(best_model)
    
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': args.max_epoch
    }, save_model_path)
    
    print("{} - Finished Training! - best Acc: {:.2f}".format(datetime.now(), best_acc*100.0))

    train_log_path = os.path.join(args.log_dir, 'train_log.txt')
    with open(train_log_path, 'w') as f:
        f.write('[Val] best acc: {}'.format(best_acc*100))
    writer.close()

if __name__ == '__main__':
    main()
