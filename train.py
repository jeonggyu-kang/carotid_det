import os, sys # /C:/Downloads path - dataset loading 
import argparse # hypermeter - user input 
import cv2           # image input/output
import numpy as np   # image
import json



import logging
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter  # tensorboard 

import Unet # Customized Unet 

def opt(): # import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epoch', type=int, default=200, help='max train epoch')

    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')

    parser.add_argument('--num_workers', type=int, default=16, help='the number of workers')
    parser.add_argument('--log_dir', type=str, default='./train_log_modular')

    parser.add_argument('--lr_drop', type=int, default=180)
    return parser.parse_args()

def test(device, model, criterion, data_loader, data_set, train_logger, writer, epoch):
        #model = model.to(device)
    model.eval()
            
    running_loss = 0.0
    running_corrects = 0

    for idx, data in enumerate(data_loader, 0):
        #* [5-2.1] input samples, labels
        train_sample, ground_truth = data
        train_sample = train_sample.to(device)
        ground_truth = ground_truth.to(device)

        with torch.set_grad_enabled(False):
            # forward pass 
            logits = model(train_sample)

        running_corrects += torch.sum(preds == ground_truth.data)

    epoch_acc = running_corrects.double() / len(data_set)
    
    writer.add_scalar('val/accuracy', epoch_acc*100.0, epoch)
    return epoch_acc


def train(device, model, optimizer, criterion, scheduler, data_loader, data_set, train_logger, writer, epoch):
    #model = model.to(device)
    model.train()
    
    running_loss = 0.0
    running_corrects = 0

    #* [5-2] singe epoch (train or validation)
    for idx, data in enumerate(data_loader, 0):
        #* [5-2.1] input samples, labels
        train_sample, ground_truth = data
        train_sample = train_sample.to(device)
        ground_truth = ground_truth.to(device)


        # set gradient to zero
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            # forward pass 
            logits = model(train_sample)
            #print(logits.shape)
            # ik, preds = torch.max(logits, 1)    ToDo
            # calcuate loss
            loss = criterion(logits, ground_truth)
            # backward pass
            loss.backward()
            # update gradient
            optimizer.step()


        #* [5-2.3] running loss and accuracy
        # statistics
        running_loss += loss.item()
        running_corrects += torch.sum(preds == ground_truth.data) # 100 80
        writer.add_scalar('loss/ce_loss', loss.item(), epoch*len(data_set) + idx)

        # print loss every 1000 steps
        if idx % 5 == 4: 
            train_logger.info("{} - [Epoch-{}][{}/{}] loss: {:.2f}".format(
                datetime.now(), epoch, idx, len(data_loader), running_loss))
            running_loss = 0.0
    scheduler.step()


    epoch_acc = running_corrects.double() / len(data_set)
    writer.add_scalar('acc', epoch_acc*100.0, epoch*len(data_set))

def main():

    model = Unet(3,2)
     
    #criterion = nn.CrossEntropyLoss()


    # optimizer
    
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
   
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)


    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[80, 160], gamma=0.1)



    train_set = CarotidDataset(....)  # ToDo


    model = model.to(device) # DDP or single gpu?
    best_acc = 0.0
    for epoch in range(start_epoch, end_epoch+1):
    
        train(device, model, optimizer, criterion, scheduler, trainloader, train_set, train_logger, writer, epoch)


        if epoch % 10 == 0: 
            validation_acc = test(device, model, criterion, testloader, test_set, train_logger, writer, epoch)
            # deep copy best model
            if validation_acc > best_acc:
                best_acc = validation_acc
                #import copy # C++ class (copy)constructor shallow copy vs deep copy
                best_model = copy.deepcopy(model.state_dict())
    
