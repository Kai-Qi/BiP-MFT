
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from functools import partial

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr
import math
from IPython import embed
import torch.nn.functional as F
from typing import List, Union, Tuple


class AttentionPool2d(nn.Module):
    """ Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.
    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py
    NOTE: This requires feature size upon construction and well prevent adaptive sizing of the network.
    """
    def __init__(
            self,
            in_features: int,
            feat_size: Union[int, Tuple[int, int]],
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 1,
            qkv_bias: bool = True,
    ):
        super().__init__()

        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.feat_size = to_2tuple(feat_size)
        self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W
        assert self.feat_size[0] == H
        assert self.feat_size[1] == W
        x = x.reshape(B, -1, N).permute(0, 2, 1)

        x = torch.cat([x.mean(1, keepdim=True), x], dim=1)
        x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = x[0], x[1], x[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        a = attn[0,0,1:16385,0].reshape(128,128)

        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, -1)
        x = self.proj(x)
        return x, a





class MLP_for_decoder(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class BaseLineSegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(BaseLineSegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        ############# seg ##############
        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP_for_decoder(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP_for_decoder(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP_for_decoder(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP_for_decoder(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.linear_pred_seg = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        ############ Classification ##############

        classes = 2
        
        self.AttentionPool2d = AttentionPool2d(in_features = 768, feat_size = [128,128])
        self.norm = nn.LayerNorm(768)

        self.linear_pred_classi_1 = nn.Linear(768, 4*768)
        self.linear_pred_classi_2 = nn.Linear(4*768, 768)
        self.linear_pred_classi_3 = nn.Linear(768, classes) 


    def forward(self, inputs):
        x = self._transform_inputs(inputs)  
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 Seg###########
        n, _, h, w = c4.shape
        new_size = c1.size()[2:]

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=new_size,mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=new_size,mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=new_size,mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1 = resize(_c1, size=new_size,mode='bilinear',align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x_for_class = x
        

        ############## MLP decoder on C1-C4 Classi###########
        x_after_bolck, a = self.AttentionPool2d(x_for_class)

        x_after_bolck_seg = x_after_bolck[:,1:,:]
        x_after_bolck_seg = x_after_bolck_seg.permute(0,2,1)
        x_after_bolck_seg = x_after_bolck_seg.reshape(x_after_bolck_seg.shape[0], 768, new_size[0], new_size[0])
        x = self.linear_pred_seg(x_after_bolck_seg)

        cls_token_final = x_after_bolck[:,0,:]
        x_classi = self.linear_pred_classi_1(cls_token_final)
        x_classi = F.gelu(x_classi)
        x_classi = self.linear_pred_classi_2(x_classi)
        x_classi = F.gelu(x_classi)
        x_classi = self.linear_pred_classi_3(x_classi)

        return x, x_classi