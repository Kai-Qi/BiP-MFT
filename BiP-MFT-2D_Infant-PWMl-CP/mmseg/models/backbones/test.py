from mix_transformer import *

import scipy.io as sio
import os
import torch


model = MixVisionTransformer(img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=True,qk_scale=None, drop_rate=0.0,
                 attn_drop_rate=0., drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                             depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]).cuda()

# data = sio.loadmat('/home/yyz/AD/data/PMWI_Data/Dataset/T1_image/01001_01.mat')['01001_01']
# data=torch.from_numpy(data)
# data1 = sio.loadmat('/home/yyz/AD/data/PMWI_Data/Dataset/T1_target/01001_08.mat')['01001_08']
# data2 = sio.loadmat('/home/yyz/AD/data/PMWI_Data/Dataset/T1_target/01001_09.mat')['01001_09']
# data3 = sio.loadmat('/home/yyz/AD/data/PMWI_Data/Dataset/T1_target/01001_16.mat')['01001_16']
x=torch.randn(1,3,512,512).cuda()


# weights = '/home/yyz/AD/PycharmProjects/weight/segformer_mit-b0_512x512_160k_ade20k_20220617_162207-c00b9603.pth'
# if os.path.exists(weights):
#     model.load_state_dict(torch.load(weights,map_location='cuda'))
#     print('successfully')
# else:
#     print('no loading')

result = model(x)
print(result.max())
print(result.min())

