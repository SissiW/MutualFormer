#!/usr/bin/python3
#coding=utf-8

import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch._six import container_abcs
from itertools import repeat
from torch.autograd import Variable
from torch.nn.parameter import Parameter

# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)

def get_S(W_adj):
    '''
    W_adj: [B, H, N, N] = torch.Size([11, 8, 100, 100])
    '''
    D = torch.pow(W_adj.sum(3).float(), -0.5)
    D = torch.diag_embed(D)
    S = (W_adj @ D).transpose(-1, -2) @ D

    # D = torch.pow(W_adj.sum(3).float(), -0.5).unsqueeze(-1)
    # D = D @ D.transpose(-1, -2)
    # S = W_adj * D
    return S

class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  
        x = x.flatten(2).transpose(1, 2) 
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MultiAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.alpha = 0.6
        self.max_iter_rd = 1
        self.max_iter_dr = 1

        self.qkv_r = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_d = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_rd = nn.Linear(dim*2, dim)
        self.proj_r = nn.Linear(dim*2, dim)
        self.proj_d = nn.Linear(dim*2, dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, xrd):
        B, N, c = x.shape   
        C = c // 2
        xr, xd = torch.split(x, C, dim=2)

        # rgb
        qkv_r = self.qkv_r(xr).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qr, kr, vr = qkv_r[0], qkv_r[1], qkv_r[2]

        attn_r = ((qr @ kr.transpose(-2, -1)) * self.scale).softmax(dim=-1) 
        S_r = get_S(attn_r)
        out_r = (attn_r @ vr).transpose(1, 2).reshape(B, N, C)

        # depth
        qkv_d = self.qkv_d(xd).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qd, kd, vd = qkv_d[0], qkv_d[1], qkv_d[2]

        attn_d = ((qd @ kd.transpose(-2, -1)) * self.scale).softmax(dim=-1) 
        S_d = get_S(attn_d) 
        out_d = (attn_d @ vd).transpose(1, 2).reshape(B, N, C)

        # initial A
        I = torch.eye(N).repeat(self.num_heads, 1, 1).repeat(B, 1, 1, 1)
        I = Parameter(I.cuda()) 
        A = I

        Y = torch.add(attn_r, attn_d)#.softmax(dim=-1)  

        # RGB -- D
        for iter in range(self.max_iter_rd):
            A_rd = self.alpha * (S_r @ A @ S_d.transpose(-1,-2)) + (1 - self.alpha) * Y  
        attn_rd = self.attn_drop(A_rd)
        out_rd = (attn_rd @ vd).transpose(1, 2).reshape(B, N, C)

        # D -- RGB
        for iter in range(self.max_iter_dr):
            A_dr = self.alpha * (S_d @ A @ S_r.transpose(-1,-2)) + (1 - self.alpha) * Y 
        attn_dr = self.attn_drop(A_dr)
        out_dr = (attn_dr @ vr).transpose(1, 2).reshape(B, N, C)

        out_r = self.proj_r(torch.cat((out_r, out_rd), dim=2))
        out_d = self.proj_d(torch.cat((out_d, out_dr), dim=2))

        qkv = self.qkv(xrd).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) 

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)  
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  
        x = self.proj_drop(self.proj(x)) + self.proj_rd(torch.cat((out_r, out_d), dim=2))

        return x


class MutualFormer(nn.Module):
    """docstring for TransBlock"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MutualFormer, self).__init__()
        self.norm1 = norm_layer(dim*2)
        self.norm1_rd = norm_layer(dim)
        self.attn = MultiAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, xrd, h, resize=False):  
        '''
        x: torch.Size([B, N, 2C])
        xrd: torch.Size([B, N, C])
        '''
        B, L, C = x.size()

        x = xrd + self.drop_path(self.attn(self.norm1(x), self.norm1_rd(xrd)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))     

        if resize:
            x = x.permute(0, 2, 1).view(B, -1, h, int(L/h))

        return x

    def initialize(self):
        self.load_state_dict(torch.load('/home/wxx/Work/reID/multi-modal-vehicle-Re-ID-master/modeling/weights/jx_vit_base_p16_224-80ecf9dd.pth'), strict=False)  

