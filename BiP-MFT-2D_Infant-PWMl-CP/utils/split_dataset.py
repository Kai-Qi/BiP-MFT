import os
from shutil import copy
import random


def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)

T1_data_path='/home/yyz/AD/data/PMWI_Data/proprosessed_data/T1/All_data/'
T1_save_path='/home/yyz/AD/data/PMWI_Data/proprosessed_data/T1/'
T2_data_path='/home/yyz/AD/data/PMWI_Data/proprosessed_data/T2/All_data/'
T2_save_path='/home/yyz/AD/data/PMWI_Data/proprosessed_data/T2/'


mkfile(T1_save_path+'train_val/'+'image')
mkfile(T1_save_path+'train_val/'+'target')
mkfile(T1_save_path+'test/'+'image')
mkfile(T1_save_path+'test/'+'target')

mkfile(T2_save_path+'train_val/'+'image')
mkfile(T2_save_path+'train_val/'+'target')
mkfile(T2_save_path+'test/'+'image')
mkfile(T2_save_path+'test/'+'target')


split_rate1 = 1-1825/2125
# split_rate2=0.5

T1_all_image_path = T1_data_path + 'image/'
T1_all_target_path=T1_data_path+'target/'
T2_all_image_path = T2_data_path + 'image/'
T2_all_target_path=T2_data_path+'target/'
T1_images = os.listdir(T1_all_image_path)
T2_images = os.listdir(T2_all_image_path)
num1 = len(T2_images)
test = random.sample(T2_images, k=int(num1 * split_rate1))
for index, image in enumerate(T1_images):
    if image in test:
        T1_image_path = T1_all_image_path + image
        T1_target_path = T1_all_target_path + image
        T1_new_image_path = T1_save_path + 'test/' + 'image/'
        T1_new_target_path = T1_save_path + 'test/' + 'target/'
        copy(T1_image_path, T1_new_image_path)
        copy(T1_target_path, T1_new_target_path)

        T2_image_path = T2_all_image_path + image
        T2_target_path = T2_all_target_path + image
        T2_new_image_path = T2_save_path + 'test/' + 'image/'
        T2_new_target_path = T2_save_path + 'test/' + 'target/'
        copy(T2_image_path, T2_new_image_path)
        copy(T2_target_path, T2_new_target_path)
    else:
        T1_image_path = T1_all_image_path + image
        T1_target_path = T1_all_target_path + image
        T1_new_image_path = T1_save_path + 'train_val/' + 'image/'
        T1_new_target_path = T1_save_path + 'train_val/' + 'target/'
        copy(T1_image_path, T1_new_image_path)
        copy(T1_target_path, T1_new_target_path)

        if image in T2_images:
            T2_image_path = T2_all_image_path + image
            T2_target_path = T2_all_target_path + image
            T2_new_image_path = T2_save_path + 'train_val/' + 'image/'
            T2_new_target_path = T2_save_path + 'train_val/' + 'target/'
            copy(T2_image_path, T2_new_image_path)
            copy(T2_target_path, T2_new_target_path)


# num2=len(val_test)
# val= random.sample(val_test, k=int(num2 * split_rate2))
# for index, image in enumerate(images):
#     if image in val_test:
#         if image in val:
#             image_path = all_image_path + image
#             target_path=all_target_path+image
#             new_image_path = save_path+'val/'+'image/'
#             new_target_path=save_path+'val/'+'target/'
#             copy(image_path, new_image_path)
#             copy(target_path, new_target_path)
#         else:
#             image_path = all_image_path + image
#             target_path = all_target_path + image
#             new_image_path = save_path + 'test/' + 'image/'
#             new_target_path = save_path + 'test/' + 'target/'
#             copy(image_path, new_image_path)
#             copy(target_path, new_target_path)
#     else:
#         image_path = all_image_path + image
#         target_path = all_target_path + image
#         new_image_path = save_path + 'train/' + 'image/'
#         new_target_path = save_path + 'train/' + 'target/'
#         copy(image_path, new_image_path)
#         copy(target_path, new_target_path)



print("processing done!")

