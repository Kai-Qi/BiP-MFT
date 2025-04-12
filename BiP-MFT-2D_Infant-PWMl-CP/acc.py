import datetime
import os

# # import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致


import torch
from encoder_decoder import *
from callbacks import LossHistory
from data_new import load_data
from losses import CE_Loss_seg,Dice_loss
from metrics import dice
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image,ImageDraw,ImageFont
from PIL import Image
from matplotlib import pylab as plt
import numpy as np
import scipy.io as sio
from metrics import *
from utils.optimizer import get_lr_scheduler, set_optimizer_lr
import argparse

torch.set_num_threads(10)


# torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser(description='Learning rate')
parser.add_argument('--learn_rate', type=float, default=0.00005, help='learn_rate')
parser.add_argument('--num_epochs', type=int, default=20, help='num_epochs')
parser.add_argument('--batch_size', type=int, default=3, help='batch_size')
parser.add_argument('--w1', type=float, default=0.1, help='w1')
parser.add_argument('--w2', type=float, default=0.7, help='w2')
parser.add_argument('--w3', type=float, default=0.2, help='w3')
parser.add_argument('--w4', type=float, default=0.1, help='w4')


parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--learn_rate2', type=float, default=0.00005, help='learn_rate')


# parser.add_argument('--loss_weight', type=float, default=1/20, help='w4')
# parser.add_argument('--loss_weight2', type=float, default=1/200, help='w4')


parser.add_argument('--seed', type=int, default=228, help='seed')


args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)



if __name__ == "__main__":

    # save_dir = '/media/datadisk/Mycode/2.My-MRI-PVL/1017/SegFormer-cross/cross_att/logs/'
    # pretrained_weight_path = '/media/datadisk/Mycode/2.My-MRI-PVL/1017/SegFormer-cross/cross_att/pretrained_weights/'
    # model_dir = '/media/datadisk/Mycode/2.My-MRI-PVL/1017/SegFormer-cross/cross_att/weights/'
    # save_results_dir = '/media/datadisk/Mycode/2.My-MRI-PVL/1017/SegFormer-cross/cross_att/results/'
    # data_path_T2 = '/media/datadisk/Mycode/2.My-MRI-PVL/1017/data/T2/'
    # data_path_T1 = '/media/datadisk/Mycode/2.My-MRI-PVL/1017/data/T1/'
    
    save_dir = '/root/no-background-dice/cross_att/logs/'
    pretrained_weight_path = '/root/pretrained_weights/'
    model_dir = '/root/no-background-dice/cross_att/weights/'
    save_results_dir = '/root/no-background-dice/cross_att/results/'
    data_path_T2 = '/root/autodl-tmp/data_to_AutoDL2/T2/'
    data_path_T1 = '/root/autodl-tmp/data_to_AutoDL2/T1/'    
    
    
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, str(time_str) + '/')
    model_path = os.path.join(model_dir, str(time_str) + '/')
    save_path = os.path.join(save_results_dir, str(time_str) + '/')
    os.mkdir(model_path)
    os.mkdir(save_path)

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes_T1     = 6
    num_classes_T2 = 2
    phi             = 'mit_b5'
    pretrained      = True

    # eval_flag           = True
    # eval_period         = 5

    input_shape  = [512, 512]
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    save_period = 50  #多少个epoch保存一次权值

    # model settings
    norm_cfg = dict(type='BN', requires_grad=True)
    find_unused_parameters = True
    
    model = EncoderDecoder_T1_T2(
        pretrained=os.path.join(pretrained_weight_path, phi+'.pth'),
        backbone=dict(
            type=phi,
            style='pytorch'),
        decode_head_T1=dict(
            type='SegFormerHead',
            in_channels=[64, 128, 320, 512],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=num_classes_T1,
            norm_cfg=norm_cfg,
            align_corners=False,
            decoder_params=dict(embed_dim=768),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        decode_head_T2=dict(
            type='BaseLineSegFormerHead',
            # in_channels=[64*2, 128*2, 320*2, 512*2],
            in_channels=[64 , 128 , 320 , 512 ],
            in_index=[0, 1, 2, 3],
            feature_strides=[4, 8, 16, 32],
            channels=128,
            dropout_ratio=0.1,
            num_classes=num_classes_T2,
            norm_cfg=norm_cfg,
            align_corners=False,
            decoder_params=dict(embed_dim=768),
            loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole'))

    weight = '/root/no-background-dice/cross_att/weights/2024_04_01_10_47_17/last_epoch_weights.pth'
    model.load_state_dict(torch.load(weight, map_location=device))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    model_train = model.train()
    cudnn.benchmark = True
    model_train = model_train.cuda()

    # optimizer
    Init_lr =  args.learn_rate # 最大学习率
    Min_lr = Init_lr * 0.001  # 最小学习率
    # gamma=0.9
    milestones=[20,80]
    T_max = 20
    eta_min = Min_lr
    # lr_decay_type = 'cos'  # 学习率下降方式

    optimizer_parameters = []

    i = 0
    j = 0
    # 遍历模型的每个参数
    for name, param in model.named_parameters():
        # 检查层名称是否包含特定关键词
        if 'eigen_weights' in name:  
            optimizer_parameters.append({'params': param, 'lr': args.learn_rate})
            i = i + 1
        else:
            optimizer_parameters.append({'params': param, 'lr': args.learn_rate2})
            j = j + 1 

    print(i)
    print(j)

    optimizer = optim.AdamW(optimizer_parameters, Init_lr, betas=(0.9, 0.999), weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=-1)


    # 数据集载入
    train_loader, val_loader, test_loader =load_data(batch_size,num_classes_T1,num_classes_T2,
                            train=True,path_T1=data_path_T1,path_T2=data_path_T2)


    CE_Loss_classi = nn.CrossEntropyLoss()
    CE_Loss_for_seg = nn.CrossEntropyLoss().cuda()


    is_test=True
    # is_test=False
    if is_test:
        print('Start Test')
        print(time_str)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_classes_T1 = 6
        num_classes_T2 = 2
        batch_size = 1


        total_score_T1 = 0
        all_total_score_T1 = torch.zeros(num_classes_T1)
        all_classes_T1 = torch.zeros(num_classes_T1)
        total_score_T2 = 0
        all_total_score_T2 = torch.zeros(num_classes_T2)
        all_classes_T2 = torch.zeros(num_classes_T2)
        total_acc_nums = 0

        model=model_train.eval()

        nums = len(test_loader)

        os.mkdir(save_path + 'T1/')
        os.mkdir(save_path + 'T2/')
        os.mkdir(save_path + 'T1/image/')
        os.mkdir(save_path + 'T1/target/')
        os.mkdir(save_path + 'T1/pre_target/')
        os.mkdir(save_path + 'T2/image/')
        os.mkdir(save_path + 'T2/target/')
        os.mkdir(save_path + 'T2/pre_target/')

        total_01 = 0
        total_10 = 0
        total_00 = 0
        total_11 = 0
        for iteration, batch in enumerate(train_loader):
            # print(iteration)
            images_T1, targets_T1, seg_labels_T1,images_T2, targets_T2, seg_labels_T2,labels,names= batch
            with torch.no_grad():
                images_T1 = images_T1.to(device)
                targets_T1 = targets_T1.to(device)
                seg_labels_T1 = seg_labels_T1.to(device)
                images_T2 = images_T2.to(device)
                targets_T2 = targets_T2.to(device)
                seg_labels_T2 = seg_labels_T2.to(device)
                labels = labels.to(device)
                labels = torch.where(labels == 2, 1, labels)    

                # 预测
                T1_seg, T2_seg, T2_seg_target_doamin, out_classi = model(images_T1, images_T2, img_metas=None)
                # dice_loss_T2_target_doamin = Dice_loss(T2_seg_target_doamin, seg_labels_T1, args.loss_weight, args.loss_weight2)
                _dice_score_T1, all_dice_score_T1, classes_T1 = dice(T1_seg, seg_labels_T1, num_classes=6)
                _dice_score_T2, all_dice_score_T2, classes_T2 = dice(T2_seg, seg_labels_T2, num_classes=2)

                result_seg_T1 = T1_seg[0]
                pr_seg_T1 = F.softmax(result_seg_T1.permute(1, 2, 0), dim=-1).detach().cpu().numpy()
                pr_seg_T1 = pr_seg_T1.argmax(axis=-1)

                tgt_T1 = np.uint8(targets_T1[0].cpu())
                tgt_T1 = Image.fromarray(tgt_T1)

                # # img.save(save_path+'image/'+str(iteration+1).zfill(3)+ '.png')
                # plt.imsave(save_path + 'T1/image/' + names[0] + '.png', images_T1[0, 0, :, :].cpu(),
                #         cmap='gray')
                # plt.imsave(save_path + 'T1/target/' + names[0] + '.png', targets_T1[0].cpu(),
                #         cmap='gray')

                # colors_T1 = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)]

                # seg_img_T1 = np.reshape(np.array(colors_T1, np.uint8)[np.reshape(pr_seg_T1, [-1])],
                #                         [pr_seg_T1.shape[0], pr_seg_T1.shape[1], -1])
                # seg_img_T1 = Image.fromarray(np.uint8(seg_img_T1))
                # draw_T1 = ImageDraw.Draw(seg_img_T1)
                # myfont = ImageFont.truetype(font='/root/no-background-dice/Ubuntu-B.ttf', size=24)
                # fillcolor = 'yellow'
                # draw_T1.text((0, 0),
                #             'dice2:' + format(round(all_dice_score_T1[1].item(), 4), '.4f') + '\ndice3:' + format(
                #                 round(all_dice_score_T1[2].item(), 4), '.4f')
                #             + '\ndice4:' + format(round(all_dice_score_T1[3].item(), 4), '.4f') + '\ndice5:' + format(
                #                 round(all_dice_score_T1[4].item(), 4), '.4f')
                #             + '\ndice6:' + format(round(all_dice_score_T1[5].item(), 4), '.4f'),
                #             font=myfont,
                #             fill=fillcolor)
                # seg_img_T1.save(save_path + 'T1/pre_target/' + names[0] + '.png')

                result_seg_T2 = T2_seg[0]
                pr_seg_T2 = F.softmax(result_seg_T2.permute(1, 2, 0), dim=-1).detach().cpu().numpy()
                pr_seg_T2 = pr_seg_T2.argmax(axis=-1)

                tgt_T2 = np.uint8(targets_T2[0].cpu())
                tgt_T2 = Image.fromarray(tgt_T2)

                # # img.save(save_path+'image/'+str(iteration+1).zfill(3)+ '.png')
                # plt.imsave(save_path + 'T2/image/' + names[0] + '.png', images_T2[0, 0, :, :].cpu(),
                #         cmap='gray')
                # plt.imsave(save_path + 'T2/target/' + names[0] + '.png', targets_T2[0].cpu(),
                #         cmap='gray')

                # colors_T2 = [(0, 0, 0), (255, 255, 255)]

                # seg_img_T2 = np.reshape(np.array(colors_T2, np.uint8)[np.reshape(pr_seg_T2, [-1])],
                #                         [pr_seg_T2.shape[0], pr_seg_T2.shape[1], -1])
                # seg_img_T2 = Image.fromarray(np.uint8(seg_img_T2))

                # draw_T2 = ImageDraw.Draw(seg_img_T2)
                # myfont = ImageFont.truetype(font='/root/no-background-dice/Ubuntu-B.ttf', size=30)
    
                
                # fillcolor = 'yellow'
                # draw_T2.text((0, 0),
                #             'dice:' + format(round(all_dice_score_T2[1].item(), 4), '.4f'),
                #             font=myfont,
                #             fill=fillcolor)
                # seg_img_T2.save(save_path + 'T2/pre_target/' + names[0] + '.png')


                _predict = torch.max(out_classi, dim=1)[1]
                # _predict = torch.where(_predict == 2, 1,_predict)     
                # labels = torch.where(labels == 2, 1, labels)     
                _acc_num=(_predict==labels).sum().item()

                total_score_T1 += _dice_score_T1.item()
                all_total_score_T1 += all_dice_score_T1.cpu()
                all_classes_T1 += classes_T1.cpu()
                total_score_T2 += _dice_score_T2.item()
                all_total_score_T2 += all_dice_score_T2.cpu()
                all_classes_T2 += classes_T2.cpu()
                total_acc_nums += _acc_num

                for i in range(T1_seg.shape[0]):
                    result_seg_T1 = T1_seg[i]
                    pr_seg_T1 = F.softmax(result_seg_T1.permute(1, 2, 0), dim=-1).detach().cpu().numpy()
                    pr_seg_T1 = pr_seg_T1.argmax(axis=-1)

                    tgt_T1 = np.uint8(targets_T1[i].cpu())
                    tgt_T1 = Image.fromarray(tgt_T1)


                    result_seg_T2 = T2_seg[i]
                    pr_seg_T2 = F.softmax(result_seg_T2.permute(1, 2, 0), dim=-1).detach().cpu().numpy()
                    pr_seg_T2 = pr_seg_T2.argmax(axis=-1)

                    # tgt_T2 = np.uint8(targets_T2[i].cpu())
                    # tgt_T2 = Image.fromarray(tgt_T2)


                    _predict = torch.max(out_classi, dim=1)[1]
                    _predict = _predict[i] 
                    _labels = labels[i]  
                    _acc_num=(_predict==_labels).sum().item()



                    t_01 = 0
                    t_10 = 0
                    t_00 = 0
                    t_11 = 0

                    if _predict == 0 and _labels == 1:    #误诊
                        t_01 = 1
                        # print('wuzhen', np.sum(pr_seg_T2))




                    if _predict == 1 and _labels == 0:    #漏诊
                        t_10 = 1
                        # print('louzhen', np.sum(pr_seg_T2))

                    if _predict == 0 and _labels == 0:    
                        t_00 = 1

                    if _predict == 1 and _labels == 1:   
                        t_11 = 1
                        
                        
                    total_01 += t_01
                    total_10 += t_10
                    total_00 += t_00
                    total_11 += t_11
                    
                    
        all_dice_T1 = all_total_score_T1 / all_classes_T1
        all_dice_T2 = all_total_score_T2 / all_classes_T2

        print('T1 mean_dice1:', all_dice_T1[0])
        print('T1 mean_dice2:', all_dice_T1[1])
        print('T1 mean_dice3:', all_dice_T1[2])
        print('T1 mean_dice4:', all_dice_T1[3])
        print('T1 mean_dice5:', all_dice_T1[4])
        print('T1 mean_dice6:', all_dice_T1[5])
        print('T2 mean_dice1:', all_dice_T2[0])
        print('T2 mean_dice2:', all_dice_T2[1])
        print('Five areas mean_dice:', (all_dice_T1[1] + all_dice_T1[2]+all_dice_T1[3]+
                                        all_dice_T1[4]+all_dice_T1[5])/5.0)
        print('acc:', total_acc_nums / nums)

        print('acc:', (total_00+total_11) /(total_01+total_10+total_00+total_11))


        print('total_Specificity:',total_11 / (total_01+total_11) )   #特异性高则意味着误诊率低
        print('total_Sensitivity:', total_00 / (total_10+total_00))   #敏感性高意味着漏诊率低

        print(total_01+total_10+total_00+total_11)
        
        print(total_01)
        print(total_10)
        print(total_00)
        print(total_11)
  

    is_acc = True

    if is_acc:
    
        model = model.eval()
        total_acc_nums = 0
        total_acc_nums_2 = 0
        total_acc_nums_3 = 0      
        total_acc_nums_4 = 0
              
        total_01 = 0
        total_10 = 0
        total_00 = 0
        total_11 = 0
        
        ttotal_01 = 0
        ttotal_10 = 0
        ttotal_00 = 0
        ttotal_11 = 0
        
        
        tttotal_01 = 0
        tttotal_10 = 0
        tttotal_00 = 0
        tttotal_11 = 0
        
        ttttotal_01 = 0
        ttttotal_10 = 0
        ttttotal_00 = 0
        ttttotal_11 = 0
        
        
        test_list = train_loader.dataset.index_total
        individual_list = []
        for i in range(len(test_list)):
            individual_list.append(test_list[i][0:5]) 
        individual_list = list(set(individual_list))

        for i in range(len(individual_list)):
            individual = individual_list[i]
            
            ##真实的个体类别
            if individual[0:2] == '01':
                individual_label = 0
            else:
                individual_label = 1
            
            label_flag = 0
            
            ##预测的个体类别
            for j in range(len(test_list)):
                
                if test_list[j][0:5]  == individual:
                    path = data_path_T2+'image_npz/' + test_list[j]
                    
                    position_T1 = data_path_T1 + 'image_npz/' + test_list[j]
                    position_T2 = data_path_T2 + 'image_npz/' + test_list[j]
           
                    image_T1=np.load(position_T1)['name']
                    image_T2 = np.load(position_T2)['name']
                    
                    
                    images_T1=np.zeros((3,image_T1.shape[0],image_T1.shape[1]),dtype='float64')
                    images_T1[0]=image_T1
                    images_T1[1]=image_T1
                    images_T1[2]=image_T1
                    images_T2=np.zeros((3,image_T2.shape[0],image_T2.shape[1]),dtype='float64')
                    images_T2[0]=image_T2
                    images_T2[1]=image_T2
                    images_T2[2]=image_T2
            

                    images_T1 = torch.Tensor(images_T1).type(torch.FloatTensor)
                    images_T2 = torch.Tensor(images_T2).type(torch.FloatTensor)
                    images_T1= torch.unsqueeze(images_T1, 0)
                    images_T2 = torch.unsqueeze(images_T2, 0)

                    with torch.no_grad():
                        images_T1 = images_T1.to(device)
                        images_T2 = images_T2.to(device)
                        
                        # 预测
                        T1_seg, T2_seg, T2_seg_target_doamin, out_classi = model(images_T1,images_T2,img_metas=None)
                        _predict = torch.max(out_classi, dim=1)[1]
                        _predict=_predict.item()
                        
                        if _predict == 0:
                            label_flag = label_flag + 1




            if individual_label == 0 and label_flag > 0:
                total_acc_nums = total_acc_nums + 1
            if individual_label == 1 and label_flag == 0:
                total_acc_nums = total_acc_nums + 1

            if individual_label == 0 and label_flag > 1:
                total_acc_nums_2 = total_acc_nums_2 + 1
            if individual_label == 1 and label_flag < 2:
                total_acc_nums_2 = total_acc_nums_2 + 1
                
            if individual_label == 0 and label_flag > 2:
                total_acc_nums_3 = total_acc_nums_3 + 1
            if individual_label == 1 and label_flag < 3:
                total_acc_nums_3 = total_acc_nums_3 + 1              
                
                
            if individual_label == 0 and label_flag > 3:
                total_acc_nums_4 = total_acc_nums_4 + 1
            if individual_label == 1 and label_flag < 4:
                total_acc_nums_4 = total_acc_nums_4 + 1                
                
                
            t_01 = 0
            t_10 = 0
            t_00 = 0
            t_11 = 0
            if individual_label == 1 and label_flag > 0:
                t_01 = t_01 + 1
            if individual_label == 0 and label_flag == 0:
                t_10 = t_10 + 1                   
            if individual_label == 0 and label_flag > 0:
                t_00 = t_00 + 1                  
            if individual_label == 1 and label_flag == 0:
                t_11 = t_11 + 1                 
            total_01 += t_01
            total_10 += t_10
            total_00 += t_00
            total_11 += t_11       
                    
                    
                    
            tt_01 = 0
            tt_10 = 0
            tt_00 = 0
            tt_11 = 0
            if individual_label == 1 and label_flag > 1:
                tt_01 = tt_01 + 1
            if individual_label == 0 and label_flag < 2:
                tt_10 = tt_10 + 1                   
            if individual_label == 0 and label_flag > 1:
                tt_00 = tt_00 + 1                  
            if individual_label == 1 and label_flag < 2:
                tt_11 = tt_11 + 1                    
            ttotal_01 += tt_01
            ttotal_10 += tt_10
            ttotal_00 += tt_00
            ttotal_11 += tt_11 
            
            
            ttt_01 = 0
            ttt_10 = 0
            ttt_00 = 0
            ttt_11 = 0
            if individual_label == 1 and label_flag > 2:
                ttt_01 = ttt_01 + 1
            if individual_label == 0 and label_flag < 3:
                ttt_10 = ttt_10 + 1                   
            if individual_label == 0 and label_flag > 2:
                ttt_00 = ttt_00 + 1                  
            if individual_label == 1 and label_flag < 3:
                ttt_11 = ttt_11 + 1                    
            tttotal_01 += ttt_01
            tttotal_10 += ttt_10
            tttotal_00 += ttt_00
            tttotal_11 += ttt_11 
            
            
            tttt_01 = 0
            tttt_10 = 0
            tttt_00 = 0
            tttt_11 = 0
            if individual_label == 1 and label_flag > 3:
                tttt_01 = tttt_01 + 1
            if individual_label == 0 and label_flag < 4:
                tttt_10 = tttt_10 + 1                   
            if individual_label == 0 and label_flag > 3:
                tttt_00 = tttt_00 + 1                  
            if individual_label == 1 and label_flag < 4:
                tttt_11 = tttt_11 + 1                    
            ttttotal_01 += tttt_01
            ttttotal_10 += tttt_10
            ttttotal_00 += tttt_00
            ttttotal_11 += tttt_11       
            
            
            
            
        print('individual acc_1:','%4f' % (total_acc_nums/len(individual_list)),
              'Specificity:', '%4f' % (total_11 / (total_01+total_11)),
              'Sensitivity:', '%4f' % (total_00 / (total_10+total_00)))
        print(total_01+total_10+total_00+total_11, ':',
              total_01, total_10, total_00, total_11)
        
        
        print('individual acc_2:','%4f' % (total_acc_nums_2/len(individual_list)),
              'Specificity:','%4f' % (ttotal_11 / (ttotal_01+ttotal_11)),
              'Sensitivity:', '%4f' % (ttotal_00 / (ttotal_10+ttotal_00)))
        print(ttotal_01+ttotal_10+ttotal_00+ttotal_11,':',
              ttotal_01, ttotal_10, ttotal_00, ttotal_11)
        
        print('individual acc_3:','%4f' % (total_acc_nums_3/len(individual_list)),
              'Specificity:','%4f' % (tttotal_11 / (tttotal_01+tttotal_11)),
              'Sensitivity:','%4f' % ( tttotal_00 / (tttotal_10+tttotal_00)))
        print(tttotal_01+tttotal_10+tttotal_00+tttotal_11,':',
              tttotal_01, tttotal_10, tttotal_00, tttotal_11)
        
        print('individual acc_4:','%4f' % (total_acc_nums_4/len(individual_list)),
              'Specificity:','%4f' % (ttttotal_11 / (ttttotal_01+ttttotal_11)),
              'Sensitivity:', '%4f' % (ttttotal_00 / (ttttotal_10+ttttotal_00)))
        print(ttttotal_01+ttttotal_10+ttttotal_00+ttttotal_11,':',
              ttttotal_01, ttttotal_10, ttttotal_00, ttttotal_11)