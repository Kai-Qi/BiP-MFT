import os

import cv2
import numpy as np
import torch
import scipy.io as sio
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader,random_split

class MyDataset(Dataset):
    def __init__(self,  num_classes, train, dataset_path):
        super(MyDataset, self).__init__()
        self.num_classes = num_classes
        self.train = train
        self.dataset_path = dataset_path
        self.name = os.listdir(os.path.join(self.dataset_path, 'image'))

    def __len__(self):
        return len(self.name)


    def __getitem__(self, index):
        image_name=self.name[index]
        image_path = os.path.join(self.dataset_path, 'image', image_name)
        target_path=os.path.join(self.dataset_path, 'target', image_name)

        # 加载mat格式数据
        image_mat = sio.loadmat(image_path)[os.path.splitext(image_name)[0]]
        target_mat=sio.loadmat(target_path)[os.path.splitext(image_name)[0]]

        # 数据增强
        image_mat, target_mat  = self.get_random_data(image_mat, target_mat, random = self.train)

        # 图像扩展为3通道
        img=np.zeros((3,image_mat.shape[0],image_mat.shape[1]),dtype='float64')
        img[0]=image_mat
        img[1]=image_mat
        img[2]=image_mat

        tgt=target_mat

        # target转化为one_hot形式
        seg_labels  = np.eye(self.num_classes)[tgt.reshape([-1])]
        seg_labels  = seg_labels.reshape(tgt.shape[0], tgt.shape[0], self.num_classes)

        return img, tgt, seg_labels,image_name

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label,random=True):
        image=Image.fromarray(np.array(image))
        label = Image.fromarray(np.array(label))

        if not random:
            return np.array(image,np.float64), np.array(label,np.uint16)

        # 翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        return np.array(image,np.float64), np.array(label,np.uint16)

def seg_dataset_collate(batch):
    images      = []
    targets        = []
    seg_labels  = []
    names = []
    for image, target, labels,image_name in batch:
        images.append(image)
        targets.append(target)
        seg_labels.append(labels)
        names.append(image_name)
    images      = torch.Tensor(images).type(torch.FloatTensor)
    targets     = torch.Tensor(targets).long()
    seg_labels  = torch.Tensor(seg_labels).type(torch.FloatTensor)
    return images, targets, seg_labels,names


def load_data(batch_size,num_classes,train,path):
    if train:
        all_dataset=MyDataset(num_classes=num_classes, train=train, dataset_path=path)
        train_dataset, val_dataset = random_split(all_dataset, [3190, 500], generator=torch.Generator().manual_seed(0))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True, drop_last=True,
                                  collate_fn=seg_dataset_collate,num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True,pin_memory=True, drop_last=True,
                                  collate_fn=seg_dataset_collate,num_workers=4)
        return train_loader,val_loader
    else:
        test_dataset=MyDataset(num_classes=num_classes, train=train, dataset_path=path)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True,pin_memory=True, drop_last=True,
                                  collate_fn=seg_dataset_collate,num_workers=4)
        return test_loader

if __name__ == '__main__':
    train_loader = load_data(batch_size=4,num_classes=6, train=True, path='/home/yyz/AD/data/PMWI_Data/proprosessed_data/T1/train_val/')

    print(len(train_loader))



