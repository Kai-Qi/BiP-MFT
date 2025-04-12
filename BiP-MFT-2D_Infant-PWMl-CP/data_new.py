import os

import cv2
import numpy as np
import torch
import scipy.io as sio
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split
import os
import numpy as np
import nibabel as nib
from  torch.utils.data import WeightedRandomSampler


class MyDataset(Dataset):
    def __init__(self,num_classes_T1,num_classes_T2,train,dataset_path,index):
        super(MyDataset, self).__init__()
        self.num_classes_T1 = num_classes_T1
        self.num_classes_T2 = num_classes_T2
        self.train = train
        
        self.slice_infos = []  

        niipath = []
        for i in range(len(index)):
            if index[i][0:2] == "01":
                niipath.append(dataset_path + "CP/" + index[i])
            if index[i][0:2] == "02":
                niipath.append(dataset_path + "N_CP/" + index[i])
            if index[i][0:2] == "03":
                niipath.append(dataset_path + "Normal/" + index[i])

        for path in niipath:
            img = nib.load(path + "/" + "case_" + path[-5:] + "_T2.nii.gz")
            data = img.get_fdata() 
            if data.ndim == 3:
                num_slices = data.shape[2]  
                for i in range(num_slices):
                    self.slice_infos.append((path, i))
                    
    def __len__(self):
        return len(self.slice_infos)

    def __getitem__(self, index):
        
        path, slice_idx = self.slice_infos[index]

        
        image_mat_T1 = nib.load(path + "/" + "case_" + path[-5:] + "_T1.nii.gz").get_fdata()[:,:,slice_idx]
        target_mat_T1= nib.load(path + "/" + "case_" + path[-5:] + "_T1_seg.nii.gz").get_fdata()[:,:,slice_idx]
        image_mat_T2 = nib.load(path + "/" + "case_" + path[-5:] + "_T2.nii.gz").get_fdata()[:,:,slice_idx]
        
        if path[-5:][1] == '3':
            target_mat_T2= np.zeros_like(image_mat_T1)
        else:
            target_mat_T2= nib.load(path + "/" + "case_" + path[-5:] + "_T2_seg.nii.gz").get_fdata()[:,:,slice_idx]
        

 
        if path[-5:][1]=='1':
            if target_mat_T2.max()>0:
                label = 0
            else:
                label = 2
        if path[-5:][1] == '2':
            if target_mat_T2.max() > 0:
                label = 1
            else:
                label = 2
        if path[-5:][1] == '3':
                label = 2


     
        image_mat_T1, target_mat_T1,image_mat_T2,target_mat_T2 = \
            self.get_random_data(image_mat_T1, target_mat_T1,image_mat_T2,target_mat_T2, random = self.train)

    
        img_T1=np.zeros((3,image_mat_T1.shape[0],image_mat_T1.shape[1]),dtype='float64')
        img_T1[0]=image_mat_T1
        img_T1[1]=image_mat_T1
        img_T1[2]=image_mat_T1
        img_T2=np.zeros((3,image_mat_T2.shape[0],image_mat_T2.shape[1]),dtype='float64')
        img_T2[0]=image_mat_T2
        img_T2[1]=image_mat_T2
        img_T2[2]=image_mat_T2

        tgt_T1=target_mat_T1
        tgt_T2 = target_mat_T2

        seg_labels_T1  = np.eye(self.num_classes_T1)[tgt_T1.reshape([-1])]
        seg_labels_T1  = seg_labels_T1.reshape(tgt_T1.shape[0], tgt_T1.shape[0], self.num_classes_T1)
        seg_labels_T2  = np.eye(self.num_classes_T2)[tgt_T2.reshape([-1])]
        seg_labels_T2  = seg_labels_T2.reshape(tgt_T2.shape[0], tgt_T2.shape[0], self.num_classes_T2)

        return img_T1,tgt_T1,seg_labels_T1,img_T2, tgt_T2,seg_labels_T2,label, path[-5:]

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image_T1, label_T1,image_T2, label_T2,random=True):
        h, w = image_T1.shape

        image_T1=Image.fromarray(np.array(image_T1))
        label_T1 = Image.fromarray(np.array(label_T1))
        image_T2=Image.fromarray(np.array(image_T2))
        label_T2 = Image.fromarray(np.array(label_T2))

        if not random:
            return np.array(image_T1,np.float64), np.array(label_T1,np.uint16),\
                   np.array(image_T2,np.float64), np.array(label_T2,np.uint16)

        
        flip = self.rand() < 0.5
        if flip:
            image_T1 = image_T1.transpose(Image.FLIP_LEFT_RIGHT)
            label_T1 = label_T1.transpose(Image.FLIP_LEFT_RIGHT)
            image_T2 = image_T2.transpose(Image.FLIP_LEFT_RIGHT)
            label_T2 = label_T2.transpose(Image.FLIP_LEFT_RIGHT)

        image_data_T1 = np.array(image_T1, np.float64)
        image_data_T2 = np.array(image_T2, np.float64)
        label_T1=np.array(label_T1, np.uint16)
        label_T2 = np.array(label_T2, np.uint16)
     
        blur = self.rand() < 0.25
        if blur:
            image_data_T1 = cv2.GaussianBlur(image_data_T1, (5, 5), 0)
            image_data_T2 = cv2.GaussianBlur(image_data_T2, (5, 5), 0)


        rotate = self.rand() < 0.25
        if rotate:
            center = (w // 2, h // 2)
            rotation = np.random.randint(-10, 11)
            M = cv2.getRotationMatrix2D(center, -rotation, scale=1)
            image_data_T1 = cv2.warpAffine(image_data_T1, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0))
            label_T1 = cv2.warpAffine(label_T1, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))
            image_data_T2 = cv2.warpAffine(image_data_T2, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=(0))
            label_T2 = cv2.warpAffine(label_T2, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0))


        return image_data_T1, label_T1,image_data_T2, label_T2

def seg_dataset_collate(batch):
    images_T1      = []
    targets_T1        = []
    seg_labels_T1 = []
    images_T2      = []
    targets_T2        = []
    seg_labels_T2 = []
    labels  = []
    names=[]

    for image_T1, target_T1, seg_label_T1,image_T2, target_T2, seg_label_T2,label,image_name in batch:
        images_T1.append(image_T1)
        targets_T1.append(target_T1)
        seg_labels_T1.append(seg_label_T1)
        images_T2.append(image_T2)
        targets_T2.append(target_T2)
        seg_labels_T2.append(seg_label_T2)
        labels.append(label)
        names.append(image_name)
    
    
    images_T1      = torch.Tensor(np.array(images_T1)).type(torch.FloatTensor)
    targets_T1     = torch.Tensor(np.array(targets_T1)/1.0).long()
    seg_labels_T1 = torch.Tensor(np.array(seg_labels_T1)).type(torch.FloatTensor)
    images_T2      = torch.Tensor(np.array(images_T2)).type(torch.FloatTensor)
    targets_T2    = torch.Tensor(np.array(targets_T2)/1.0).long()
    seg_labels_T2 = torch.Tensor(np.array(seg_labels_T2)).type(torch.FloatTensor)   
    
    
    
    labels  = torch.Tensor(labels).long()
    return images_T1, targets_T1, seg_labels_T1,images_T2, targets_T2, seg_labels_T2,labels,names


