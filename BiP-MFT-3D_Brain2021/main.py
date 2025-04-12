
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import argparse

from timeit import default_timer
import numpy as np
import random
import torch
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from BraTS import get_datasets
from architectures.BiP_MFT_3D import BiP_MFT_3D
from models import DataAugmenter
from utils import mkdir, save_best_model, save_seg_csv, cal_dice, cal_confuse, save_test_label, AverageMeter, save_checkpoint
from torch.backends import cudnn
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.losses.dice import DiceLoss
from monai.inferers import sliding_window_inference
import datetime
from functools import reduce
from functools import partial
import operator
import torch.nn as nn
torch.set_num_threads(10)


global time_str
time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
print(time_str)


path = "/data02/qikai/BestResult_Ours-SegFormer3D-toAutoDL/Ours-SegFormer3D-toAutoDL/"


parser = argparse.ArgumentParser(description='BraTS')
parser.add_argument('--exp-name', default="CKD", type=str)
parser.add_argument('--mode', choices=['train', 'test'], default='train')

parser.add_argument('--dataset-folder',default="/data02/qikai/CKD-TransBTS-Total/CKD-TransBTS-main/dataset_partial_data/", type=str, help="Please reference the README file for the detailed dataset structure.")
parser.add_argument('--workers', default=4, type=int, help="The value of CPU's num_worker")
parser.add_argument('--end-epoch', default=2, type=int, help="Maximum iterations of the model")
parser.add_argument('--batch-size', default=1, type=int)
parser.add_argument('--lr', default=1e-6, type=float) 
parser.add_argument('--devices', default=0, type=int)
parser.add_argument('--resume', default=False, type=bool)
parser.add_argument('--tta', default=True, type=bool, help="test time augmentation")
parser.add_argument('--seed', default=1)
parser.add_argument('--val', default=1, type=int, help="Validation frequency of the model")
parser.add_argument('--test', default=100, type=int, help="test frequency of the model")
parser.add_argument('--label_folder', default= path + 'test_save_label_folder', type=str)
parser.add_argument('--writer_folder', default= path + 'writer', type=str)
parser.add_argument('--csv_folder', default= path + 'csv', type=str)
parser.add_argument('--best_folder', default= path + 'best_model', type=str)
parser.add_argument('--drop', default=0.2, type=float, help="test frequency of the model")
parser.add_argument('--learn_rate2', type=float, default=1e-6, help='learn_rate')
parser.add_argument('--eta_min', default=1e-7, type=float) 
parser.add_argument('--clip', default=310, type=float) 
parser.add_argument('--drop_path_rate', default=0.1, type=float, help="test frequency of the model")


def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

def init_randon(seed):
    torch.manual_seed(seed)        
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    cudnn.benchmark = False         
    cudnn.deterministic = True

def init_folder(args):
        
    args.base_folder =  mkdir(os.path.dirname(os.path.realpath(__file__)))
    args.dataset_folder = mkdir(os.path.join(args.base_folder, args.dataset_folder))
    
    args.best_folder = mkdir(os.path.join(args.base_folder+ '/'+'best_model'+ '/'+args.exp_name + '/'+str(time_str) + '/'))
    args.writer_folder = mkdir(os.path.join(args.base_folder+ '/'+'writer'+ '/'+args.exp_name + '/'+str(time_str) + '/'))
    args.pred_folder = mkdir(os.path.join(args.base_folder+ '/'+'pred'+ '/'+args.exp_name + '/'+str(time_str) + '/'))
    args.checkpoint_folder = mkdir(os.path.join(args.base_folder+ '/'+'checkpoint'+ '/'+args.exp_name + '/'+str(time_str) + '/'))
    args.csv_folder = mkdir(os.path.join(args.base_folder+ '/'+'csv'+ '/'+args.exp_name + '/'+str(time_str) + '/'))

    print(f"The code folder are located in {os.path.dirname(os.path.realpath(__file__))}")
    print(f"The dataset folder located in {args.dataset_folder}")
    
def main(args):  
    writer = SummaryWriter(args.writer_folder)
    
    model = BiP_MFT_3D(in_channels = 4,
        sr_ratios = [8, 4, 2, 1],
        embed_dims = [64, 128, 320, 512],
        depths = [3, 4, 6, 3],
        decoder_dropout = args.drop,
        drop_path_rate = args.drop_path_rate).cuda()
    
    criterion=DiceLoss(sigmoid=True).cuda()
    optimizer_parameters = []

    for name, param in model.named_parameters():
        if 'eigen_weights' in name:  
            optimizer_parameters.append({'params': param, 'lr': args.lr})
        else:
            optimizer_parameters.append({'params': param, 'lr': args.learn_rate2})
    
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
    print(count_params(model))

    if args.mode == "train":
        train_dataset = get_datasets(args.dataset_folder, "train")
        train_val_dataset = get_datasets(args.dataset_folder, "train_val")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True, pin_memory=True)
        train_val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)
        
        test_dataset = get_datasets(args.dataset_folder, "test")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=args.workers,pin_memory=True)
        train_manager(args, train_loader, train_val_loader, test_loader, model, criterion, optimizer, writer)
        
        print("start test")
        model.load_state_dict(torch.load(os.path.join(args.best_folder, "best_model.pkl")))
        model.eval()

        test(args, "test", test_loader, model)
        
    elif args.mode == "test" :
        print("start test")
        model.load_state_dict(torch.load(os.path.join(args.best_folder, "best_model.pkl")))
        model.eval()
        test_dataset = get_datasets(args.dataset_folder, "test")
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=args.workers,pin_memory=True)
        test(args, "test", test_loader, model)
    

def train_manager(args, train_loader, train_val_loader, test_loader, model, criterion, optimizer, writer):
    best_loss = np.inf
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.end_epoch, eta_min= args.eta_min )

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(os.path.join(args.checkpoint_folder, "checkpoint.pth.tar"))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])
    print(f"start train from epoch = {start_epoch}")
    
    train_l2_record = []
    test_l2_record = []    
    
    for epoch in range(start_epoch, args.end_epoch):
        t1 = default_timer()
        model.train()
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        train_loss, norm10 = train(train_loader, model, criterion, optimizer, scheduler, epoch, writer)
        train_val_loss = []
        if epoch < 400:
            if (epoch + 1) % args.val == 0:
                model.eval()
                with torch.no_grad():
                    train_val_loss = train_val(train_val_loader, model, criterion, epoch, writer)
                    if train_val_loss < best_loss:
                        best_loss = train_val_loss
                        save_best_model(args, model)
        if epoch > 400:
            model.eval()
            with torch.no_grad():
                train_val_loss = train_val(train_val_loader, model, criterion, epoch, writer)
                if train_val_loss < best_loss:
                    best_loss = train_val_loss
                    save_best_model(args, model)     
                    
                    
        save_checkpoint(args, dict(epoch=epoch, model = model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict()))
        t2 = default_timer()
        print(f"epoch = {epoch},  time = {'%.2f'% (t2-t1)}, train_loss = {'%.4f'% (train_loss)}, train_val_loss = {train_val_loss}, best_loss = {best_loss}, grad_norm = {'%.4f'% max(norm10)},  grad_norm_max = {'%.4f'% max(norm10)}, grad_norm_mean = {'%.4f'% (sum(norm10)/len(norm10))} ")            
            
        train_l2_record.append(train_loss)
        test_l2_record.append(train_val_loss)
    
    
    import scipy.io as io
    io.savemat(args.best_folder + 'train_process.mat', 
            {'train_loss': np.array(train_l2_record, dtype=object), 'train_val_loss': np.array(test_l2_record, dtype=object)})      
        
    print("finish train epoch")

def train(data_loader, model, criterion, optimizer, scheduler, epoch, writer):
    train_loss_meter = AverageMeter('Loss', ':.4e')
    norm10 = []
    

    for i, data in enumerate(data_loader):
        # torch.cuda.empty_cache()
        data_aug = DataAugmenter().cuda()
        label = data["label"].cuda()
        images = data["image"].cuda()
        images, label = data_aug(images, label)
        pred = model(images)
        train_loss = criterion(pred, label)
        train_loss_meter.update(train_loss.item())
       
        train_loss.backward()
        norm10.append( nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip, norm_type=2) )

        optimizer.step()
    scheduler.step()
    # torch.cuda.empty_cache()
    writer.add_scalar(path + "loss/train", train_loss_meter.avg, epoch)
    return train_loss_meter.avg, norm10


def train_val(data_loader, model, criterion, epoch, writer):
    train_val_loss_meter = AverageMeter('Loss', ':.4e')
    for i, data in enumerate(data_loader):
        label = data["label"].cuda()
        images = data["image"].cuda()
        pred = model(images)
        train_val_loss = criterion(pred, label)
        train_val_loss_meter.update(train_val_loss.item())
    writer.add_scalar(path + "loss/train_val", train_val_loss_meter.avg, epoch)
    return train_val_loss_meter.avg

def inference(model, input, batch_size=2, overlap=0.6):
    def _compute(input):
        return sliding_window_inference(inputs=input, roi_size=(128, 128, 128), sw_batch_size=batch_size, predictor=model, overlap=overlap)
    return _compute(input)


def test(args, mode, data_loader, model):
    
    metrics_dict = []
    haussdor = HausdorffDistanceMetric(include_background=True, percentile=95)
    meandice = DiceMetric(include_background=True)
    for i, data in enumerate(data_loader):
        patient_id = data["patient_id"][0]
        inputs = data["image"]
        targets = data["label"].cuda()
        pad_list = data["pad_list"]
        nonzero_indexes = data["nonzero_indexes"]
        inputs = inputs.cuda()
        model.cuda()
        with torch.no_grad():  
            if args.tta:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2,)).flip(dims=(2,)), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(3,)).flip(dims=(3,)), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(4,)).flip(dims=(4,)), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 3)).flip(dims=(2, 3)), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 4)).flip(dims=(2, 4)), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(3, 4)).flip(dims=(3, 4)), batch_size=2, overlap=0.6))
                predict += torch.sigmoid(inference(model, inputs.flip(dims=(2, 3, 4)).flip(dims=(2, 3, 4)), batch_size=2, overlap=0.6))
                predict = predict / 8.0 
            else:
                predict = torch.sigmoid(inference(model, inputs, batch_size=2, overlap=0.6))
                
        targets = targets[:, :, pad_list[-4]:targets.shape[2]-pad_list[-3], pad_list[-6]:targets.shape[3]-pad_list[-5], pad_list[-8]:targets.shape[4]-pad_list[-7]]
        predict = predict[:, :, pad_list[-4]:predict.shape[2]-pad_list[-3], pad_list[-6]:predict.shape[3]-pad_list[-5], pad_list[-8]:predict.shape[4]-pad_list[-7]]
        predict = (predict>0.5).squeeze()
        targets = targets.squeeze()
        dice_metrics = cal_dice(predict, targets, haussdor, meandice)
        confuse_metric = cal_confuse(predict, targets, patient_id)
        et_dice, tc_dice, wt_dice = dice_metrics[0], dice_metrics[1], dice_metrics[2]
        et_hd, tc_hd, wt_hd = dice_metrics[3], dice_metrics[4], dice_metrics[5]
        et_sens, tc_sens, wt_sens = confuse_metric[0][0], confuse_metric[1][0], confuse_metric[2][0]
        et_spec, tc_spec, wt_spec = confuse_metric[0][1], confuse_metric[1][1], confuse_metric[2][1]
        metrics_dict.append(dict(id=patient_id,
            et_dice=et_dice, tc_dice=tc_dice, wt_dice=wt_dice, 
            et_hd=et_hd, tc_hd=tc_hd, wt_hd=wt_hd,
            et_sens=et_sens, tc_sens=tc_sens, wt_sens=wt_sens,
            et_spec=et_spec, tc_spec=tc_spec, wt_spec=wt_spec))
        full_predict = np.zeros((155, 240, 240))
        predict = reconstruct_label(predict)
        full_predict[slice(*nonzero_indexes[0]), slice(*nonzero_indexes[1]), slice(*nonzero_indexes[2])] = predict
        # save_test_label(args, patient_id, full_predict)
    save_seg_csv(args, mode, metrics_dict)
    import pandas as pd

    df = pd.DataFrame(metrics_dict)
    mean_values = df.drop(columns='id').mean()
    print(mean_values)
    print(time_str)
    print( (mean_values['et_dice'] + mean_values['tc_dice'] + mean_values['wt_dice'])/3.0)
    print( (mean_values['et_hd'] + mean_values['tc_hd'] + mean_values['wt_hd'])/3.0)
    print( (mean_values['et_sens'] + mean_values['tc_sens'] + mean_values['wt_sens'])/3.0)


def reconstruct_label(image):
    if type(image) == torch.Tensor:
        image = image.cpu().numpy()
    c1, c2, c3 = image[0], image[1], image[2]
    image = (c3 > 0).astype(np.uint8)
    image[(c2 == False)*(c3 == True)] = 2
    image[(c1 == True)*(c3 == True)] = 4
    return image

if __name__=='__main__':
    args=parser.parse_args()
    for arg in vars(args):
        print(format(arg, '<20'), format(str(getattr(args, arg)), '<'))
    if torch.cuda.device_count() == 0:
        raise RuntimeWarning("Can not run without GPUs")
    init_randon(args.seed)
    init_folder(args)
    torch.cuda.set_device(args.devices)
    main(args)

