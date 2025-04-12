import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from metrics import *


def CE_Loss_seg(inputs, target, num_classes=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss()(temp_inputs, temp_target)
    return CE_loss

def CE_Loss_classi(inputs, target):

    CE_loss  = nn.CrossEntropyLoss()(inputs, target)
    
    return CE_loss

def Dice_loss(inputs, target,  beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # 计算dice loss
    tp = torch.sum(temp_target * temp_inputs, axis=[1])
    fp = torch.sum(temp_inputs, axis=[1]) - tp
    fn = torch.sum(temp_target, axis=[1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    
    class_target = torch.sum(temp_target, dim = 1)
    class_target = class_target > 0

    result = score * class_target
    
    if result.shape[1] == 6:
        weight = torch.tensor([0.0,1,1,1,1,1]).cuda()
        result = result * weight
    
    if result.shape[1] == 2:
        weight = torch.tensor([0.0,1]).cuda()
        result = result * weight
        
        
    dice_loss = 1 - torch.sum(result) / (torch.count_nonzero(result) + 1e-4)

    
    
    return dice_loss



def T1_Dice_loss_right(inputs, target):

    batchsize = inputs.shape[0]

    # n, c, h, w = inputs.size()
    # nt, ht, wt, ct = target.size()
    # if h != ht and w != wt:
    #     inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    pre_total = []
    labels_total = []
    T1_pre = torch.argmax(inputs, dim=1)

    for i in range(batchsize):
        pre_total.append(T1_pre[i,:,:].cpu().numpy())
        labels_total.append(target[i,:,:].cpu().numpy())

    T_Dice = eval_metrics(results=pre_total, gt_seg_maps=labels_total,
                            num_classes=6, ignore_index=7, metrics=['mDice'])
    
    return T_Dice

def T2_Dice_loss_right(inputs, target):

    batchsize = inputs.shape[0]

    # n, c, h, w = inputs.size()
    # nt, ht, wt, ct = target.size()
    # if h != ht and w != wt:
    #     inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    pre_total = []
    labels_total = []
    T1_pre = torch.argmax(inputs, dim=1)

    for i in range(batchsize):
        pre_total.append(T1_pre[i,:,:].cpu().numpy())
        labels_total.append(target[i,:,:].cpu().numpy())

    T_Dice = eval_metrics(results=pre_total, gt_seg_maps=labels_total,
                            num_classes=2, ignore_index=3, metrics=['mDice'])
    
    return T_Dice