# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed

class MLP(nn.Module):
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
class ClassiHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(ClassiHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim_classi = 256
        classes=2
        self.linear_c4_classi = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim_classi)
        self.linear_c3_classi = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim_classi)
        self.linear_c2_classi = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim_classi)
        self.linear_c1_classi = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim_classi)

        self.linear_fuse_classi = ConvModule(
            in_channels=embedding_dim_classi*4,
            out_channels=embedding_dim_classi,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='BN', requires_grad=True)
        )

        self.pred_classi = nn.Conv2d(embedding_dim_classi, 32, kernel_size=1)
        self.linear_pred_classi = nn.Linear(32*16*16, classes)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        n, _, h, w = c4.shape

        ############## MLP decoder on C1-C4 Classi###########
        _c4_classi = self.linear_c4_classi(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])

        _c3_classi = self.linear_c3_classi(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3_classi = resize(_c3_classi, size=c4.size()[2:], mode='bilinear', align_corners=False)

        _c2_classi = self.linear_c2_classi(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2_classi = resize(_c2_classi, size=c4.size()[2:], mode='bilinear', align_corners=False)

        _c1_classi = self.linear_c1_classi(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])
        _c1_classi = resize(_c1_classi, size=c4.size()[2:], mode='bilinear', align_corners=False)

        _c_classi = self.linear_fuse_classi(torch.cat([_c4_classi, _c3_classi, _c2_classi, _c1_classi], dim=1))

        x_classi = self.dropout(_c_classi)
        x_classi=self.pred_classi(x_classi)
        x_classi = x_classi.flatten(1)
        x_classi=self.linear_pred_classi(x_classi)

        return x_classi
