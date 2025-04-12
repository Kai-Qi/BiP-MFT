import torch
import torch.nn.functional as F


def dice_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)
     # 计算dice系数
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target, axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    mean_score = torch.mean(score)
    return mean_score,score

def dice_score_test(inputs, target,num_classes=6, beta=1, smooth=1e-5, threhold=0.5):

    classes=torch.zeros(num_classes)

    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)
     # 计算dice系数
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1])
    fn = torch.sum(temp_target, axis=[0, 1])
    score = ((1 + beta ** 2) * tp + smooth) / (beta ** 2 * fn + fp + smooth)
    for i in range(num_classes):
        if fn[i]==0:
            score[i]=0
        else:
            classes[i]=1
    mean_score = torch.sum(score)/torch.sum(classes)
    return mean_score,score,classes


def binary_dice(predict, target,smooth=1e-5):
    num = torch.sum(torch.mul(predict, target), dim=1)
    gt_num = torch.sum(target, dim=1)
    pre_num = torch.sum(predict, dim=1)
    score = (2 * num + smooth) / ( gt_num + pre_num + smooth)
    exit_num=0
    for i in range(predict.shape[0]):
        if gt_num[i]==0:
            score[i]=0
        else:
            exit_num+=1
    # if exit_num==0:
    #     mean_score=0
    # else:
    #     mean_score = torch.sum(score)/exit_num
    return torch.sum(score),exit_num


def dice(predict, target,num_classes=6,threhold=0.5):

    total_dice = 0
    classes_list = torch.zeros(num_classes)
    dice_list=torch.zeros(num_classes)

    n, c, h, w = predict.size()
    nt, ht, wt, ct = target.size()
    temp_inputs = predict.contiguous().view(n,c,-1)
    temp_target = target.transpose(1, 2).transpose(1, 3).view(n, ct, -1)
    predict = F.softmax(temp_inputs, dim=1)
    predict = torch.gt(predict, threhold).float()

    for i in range(num_classes):
        dice_score,class_num = binary_dice(predict[:, i], temp_target[:, i])
        dice_list[i]=dice_score
        classes_list[i]=class_num
        total_dice+=dice_score


    mean_dice=total_dice/torch.sum(classes_list)

    return mean_dice,dice_list,classes_list