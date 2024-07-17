# YOLOv5 common modules

import math
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F

from utils.datasets import letterbox
from utils.general import non_max_suppression, make_divisible, scale_coords, increment_path, xyxy2xywh, save_one_box
from utils.plots import colors, plot_one_box
from utils.torch_utils import time_synchronized
import torch.fft as fft
from torch.nn import init, Sequential


# 定义傅里叶变换函数
def fourier_transform(image):
    # 将图像转换为频域表示
    image_freq = fft.fftn(image, dim=(-2, -1))

    # 将频域图像移动零频率到中心
    image_freq_shifted = fft.fftshift(image_freq, dim=(-2, -1))

    return image_freq_shifted

# 提取低频和高频图像


def extract_frequency2(image):
    # 进行二维傅里叶变换
    f = fft.fftn(image, dim=(-2, -1))
    f_shift = fft.fftshift(f, dim=(-2, -1))

    # 将频域图像分离为高频和低频成分
    batch_size, num_channels, rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2  # 频域图像的中心点

    # 选择阈值来分离高频和低频成分
    threshold = crow + ccol // 4  # 可根据需要调整阈值
    f_shift_highpass = f_shift.clone()
    f_shift_highpass[:, :, crow - threshold:crow +
                     threshold, ccol - threshold:ccol + threshold] = 0

    f_shift_lowpass = f_shift.clone()
    f_shift_lowpass[:, :, :crow - threshold, :] = 0
    f_shift_lowpass[:, :, crow + threshold:, :] = 0
    f_shift_lowpass[:, :, :, :ccol - threshold] = 0
    f_shift_lowpass[:, :, :, ccol + threshold:] = 0

    # 进行逆傅里叶变换，得到分离后的高频图像和低频图像
    f_highpass = fft.ifftshift(f_shift_highpass, dim=(-2, -1))
    image_highpass = fft.ifftn(f_highpass, dim=(-2, -1))

    f_lowpass = fft.ifftshift(f_shift_lowpass, dim=(-2, -1))
    image_lowpass = fft.ifftn(f_lowpass, dim=(-2, -1))

    # 将结果转换为半精度浮点数
    image_lowpass = image_lowpass.half()
    image_highpass = image_highpass.half()

    return image_lowpass, image_highpass


def extract_frequency(image):
    # 进行傅里叶变换
    # image = image.to(torch.float32)
    image_freq = fourier_transform(image)

    # 获取图像的中心位置
    _, _, H, W = image.size()
    center_h, center_w = H // 2, W // 2

    # 选择阈值来分离高频和低频成分
    threshold = 30  # 可根据需要调整阈值

    # 低频成分
    image_low_freq = image_freq.clone()
    image_low_freq[:, :, center_h - threshold:center_h +
                   threshold, center_w - threshold:center_w + threshold] = 0

    # 高频成分
    image_high_freq = image_freq - image_low_freq

    # image_low_freq, image_high_freq
    return image_low_freq.half(), image_high_freq.half()


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


def DWConv(c1, c2, k=1, s=1, act=True):
    # Depthwise convolution
    return Conv(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Conv, self).__init__()
        # print(c1, c2, k, s,)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        # print("Conv", x.shape)
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def Seperation_loss(M):
    l = M.size(0)
    device = M.device
    L_sep = torch.tensor(0.0, device=device)

    for i in range(l - 1):
        for j in range(i + 1, l):
            matrix = M[i].t() @ M[j]
            L_sep += matrix

    L_sep = L_sep / (l * (l - 1))
    return L_sep


class GPT1(nn.Module):
    """  the full GPT1 language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model  # 128
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(
            1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d(
            (self.vert_anchors, self.horz_anchors))

        # convolution with kernel size 1
        self.conv1 = nn.Conv2d(
            d_model, 8, kernel_size=1, stride=1, padding=0, bias=False)  # d_model, d_model
        self.sig = nn.Sigmoid()
        # convolution with kernel size 1
        self.conv2 = nn.Conv2d(
            8, d_model, kernel_size=1, stride=1, padding=0, bias=False)  # d_model, d_model

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        self.pattenLoss = 0
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape
        # print("bs:{}, c:{}, h:{}, w:{}".format(bs, c, h, w)) # bs:4, c:128, h:160, w:160

        # bs, c, h, w = rgb_fea.shape
        # print("bs:{}, c:{}, h:{}, w:{}".format(bs, c, h, w)) # bs:4, c:128, h:160, w:160
        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)
        # print("rgb_fea.shape:{}".format(rgb_fea.shape)) # rgb_fea.shape:torch.Size([4, 128, 8, 8])
        # print("ir_fea.shape:{}".format(ir_fea.shape))  # ir_fea.shape:torch.Size([4, 128, 8, 8])

        ###########################################################
        # rgb_fea_patten = rgb_fea.view(-1, h, w)
        # ir_fea_patten = ir_fea.view(-1, h, w)
        bs_pool, c_pool, h_pool, w_pool = rgb_fea.shape
        rgb_fea_patten = rgb_fea
        ir_fea_patten = ir_fea
        rgb_fea_patten_conv = self.conv1(rgb_fea_patten)
        rgb_fea_patten_M = self.sig(rgb_fea_patten_conv)

        ir_fea_patten_conv = self.conv1(ir_fea_patten)
        ir_fea_patten_M = self.sig(ir_fea_patten_conv)

        rgb_fea_patten_reshape = rgb_fea_patten_M.view(-1, h_pool * w_pool)
        ir_fea_patte_reshape = ir_fea_patten_M.view(-1, h_pool * w_pool)

        concatenated_fea_patteen = torch.cat(
            (rgb_fea_patten_reshape, ir_fea_patte_reshape), dim=0)
        # L_rgb = loss_function(rgb_fea_patten_reshape)
        # L_ir = loss_function(ir_fea_patte_reshape)
        self.pattenLoss = Seperation_loss(concatenated_fea_patteen)
        # self.pattenLoss =  L_rgb + L_ir
        # print("self.pattenLoss:{}".format(self.pattenLoss))

        rgb_fea_PT = self.conv2(rgb_fea_patten_M)
        ir_fea_PT = self.conv2(ir_fea_patten_M)

        P_rgb_fea = rgb_fea_PT * rgb_fea
        P_ir_fea = ir_fea_PT * ir_fea

        rgb_fea_flat = P_rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = P_ir_fea.view(bs, c, -1)  # flatten the feature
        ###########################################################

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # # pad token embeddings along number of tokens dimension
        # rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        # ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature

        # print("rgb_fea_flat.shape:{}".format(rgb_fea_flat.shape)) # rgb_fea_flat.shape:torch.Size([4, 128, 64])
        # print("ir_fea_flat.shape:{}".format(ir_fea_flat.shape)) # ir_fea_flat.shape:torch.Size([4, 128, 64])

        token_embeddings = torch.cat(
            [rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        # print("token_embeddings.shape:{}".format(token_embeddings.shape)) # token_embeddings.shape: torch.Size([4, 128, 128])

        token_embeddings = token_embeddings.permute(
            0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)
        # print("Permute token_embeddings.shape:{}".format(token_embeddings.shape)) #  Permute token_embeddings.shape: torch.Size([4, 128, 128])

        # transformer
        # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.drop(self.pos_emb + token_embeddings)
        # print("Pre Transfomer x.shape:{}".format(x.shape)) #  Pre Transfomer x.shape:torch.Size([4, 128, 128])

        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(
            bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(
            bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(
            rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out, self.pattenLoss


class GPT1_fourier(nn.Module):
    """  the full GPT1_fourier language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model  # 128
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(
            1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d(
            (self.vert_anchors, self.horz_anchors))

        # convolution with kernel size 1
        self.conv1 = nn.Conv2d(
            d_model, 8, kernel_size=1, stride=1, padding=0, bias=False)  # d_model, d_model
        self.sig = nn.Sigmoid()
        # convolution with kernel size 1
        self.conv2 = nn.Conv2d(
            8, d_model, kernel_size=1, stride=1, padding=0, bias=False)  # d_model, d_model

        # ######### heatmap
        # # 定义转置卷积层进行反向池化操作
        # self.transposed_conv = nn.ConvTranspose2d(in_channels = 128, out_channels = 128, kernel_size = 16, stride = 2, padding = 36)

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)

        # ###

        # rgb_fea_CPU = rgb_fea.cpu()
        # ir_fea_CPU  = ir_fea.cpu()
        # heatmap_rgb_fea_CPU = rgb_fea_CPU.detach().numpy()[0][0]
        # heatmap_ir_fea_CPU  = ir_fea_CPU.detach().numpy()[0][0]
        # print("rgb_fea shape:{} , heatmap_rgb_fea_CPU.shape:{}".format(rgb_fea.shape, heatmap_rgb_fea_CPU.shape))
        # # 绘制heatmap
        # plt.imshow(heatmap_rgb_fea_CPU, cmap='viridis', interpolation='nearest')
        # # 保存图像到本地
        # plt.savefig('heatmap_rgb_fea_CPU.png')

        # # 绘制heatmap
        # plt.imshow(heatmap_ir_fea_CPU, cmap='viridis', interpolation='nearest')
        # # 保存图像到本地
        # plt.savefig('heatmap_ir_fea_CPU.png')
        # ###

        self.pattenLoss = torch.tensor(0.0, device=x[0].device)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape
        # print("bs:{}, c:{}, h:{}, w:{}".format(bs, c, h, w)) # bs:4, c:128, h:160, w:160

        # bs, c, h, w = rgb_fea.shape
        # print("bs:{}, c:{}, h:{}, w:{}".format(bs, c, h, w)) # bs:4, c:128, h:160, w:160
        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)
        # print("rgb_fea.shape:{}".format(rgb_fea.shape)) # rgb_fea.shape:torch.Size([4, 128, 8, 8])
        # print("ir_fea.shape:{}".format(ir_fea.shape))  # ir_fea.shape:torch.Size([4, 128, 8, 8])

        ###########################################################
        # rgb_fea_patten = rgb_fea.view(-1, h, w)
        # ir_fea_patten = ir_fea.view(-1, h, w)
        bs_pool, c_pool, h_pool, w_pool = rgb_fea.shape
        rgb_fea_patten = rgb_fea
        ir_fea_patten = ir_fea

        #############################
        rgb_fea_patten_low, rgb_fea_patten_high = extract_frequency2(rgb_fea)
        ir_fea_patten_low, ir_fea_patten_high = extract_frequency2(ir_fea)

        # ###
        # print("rgb_fea_patten_high.shape:", rgb_fea_patten_high.shape)
        # transposed_rgb_fea_patten_high = torch.nn.functional.interpolate(rgb_fea_patten_high, size=(120, 160), mode='bilinear', align_corners=False)
        # transposed_ir_fea_patten_high = torch.nn.functional.interpolate(ir_fea_patten_high, size=(120, 160), mode='bilinear', align_corners=False)
        # result_transposed_rgb_fea_patten_high = torch.mul(transposed_rgb_fea_patten_high, x[0])
        # result_transposed_ir_fea_patten_high = torch.mul(transposed_ir_fea_patten_high, x[1])

        # rgb_fea_patten_high_CPU = result_transposed_rgb_fea_patten_high.cpu()
        # ir_fea_patten_high_CPU = result_transposed_ir_fea_patten_high.cpu()
        # heatmap_rgb_high = rgb_fea_patten_high_CPU.detach().numpy()[0][0]
        # heatmap_ir_high  = ir_fea_patten_high_CPU.detach().numpy()[0][0]

        # # 绘制heatmap
        # plt.imshow(heatmap_rgb_high, cmap='viridis', interpolation='nearest')
        # # 保存图像到本地
        # plt.savefig('heatmap_rgb_high_multiple.png')

        # # 绘制heatmap
        # plt.imshow(heatmap_ir_high, cmap='viridis', interpolation='nearest')
        # # 保存图像到本地
        # plt.savefig('heatmap_ir_high_multiple.png')
        ###

        # rgb_fea_patten_low_conv = self.conv1(rgb_fea_patten_low) # rgb_fea_patten_low
        # rgb_fea_patten_low_M = self.sig(rgb_fea_patten_low_conv)
        # rgb_fea_patten_low_reshape = rgb_fea_patten_low_M.view(-1, h_pool * w_pool)
        ###

        ###
        rgb_fea_patten_high_multi = torch.mul(rgb_fea_patten_high, rgb_fea)
        ir_fea_patten_high_multi = torch.mul(ir_fea_patten_high, ir_fea)

        ###
        rgb_fea_patten_high_conv = self.conv1(
            rgb_fea_patten_high_multi)  # rgb_fea_patten_high
        rgb_fea_patten_high_M = self.sig(rgb_fea_patten_high_conv)
        rgb_fea_patten_high_reshape = rgb_fea_patten_high_M.view(
            -1, h_pool * w_pool)


        ir_fea_patten_high_conv = self.conv1(
            ir_fea_patten_high_multi)  # ir_fea_patten_high
        ir_fea_patten_high_M = self.sig(ir_fea_patten_high_conv)
        ir_fea_patten_high_reshape = ir_fea_patten_high_M.view(
            -1, h_pool * w_pool)

        # rgb_fea_patten_high_conv = self.conv1(rgb_fea_patten_high)

        # rgb_fea_patten_M = self.sig(rgb_fea_patten_conv)

        # ir_fea_patten_conv  = self.conv1(ir_fea_patten)
        # ir_fea_patten_M = self.sig(ir_fea_patten_conv)

        # rgb_fea_patten_reshape = rgb_fea_patten_M.view(-1, h_pool * w_pool)
        # ir_fea_patte_reshape = ir_fea_patten_M.view(-1, h_pool * w_pool)

        # concatenated_fea_patteen = torch.cat((rgb_fea_patten_reshape, ir_fea_patte_reshape), dim=0)

        #############################

        ###
        rgb_fea_patten_multiply = torch.mul(rgb_fea_patten, rgb_fea)
        ir_fea_patten_multiply = torch.mul(ir_fea_patten, rgb_fea)
        ###

        rgb_fea_patten_conv = self.conv1(rgb_fea_patten)
        rgb_fea_patten_M = self.sig(rgb_fea_patten_conv)

        ir_fea_patten_conv = self.conv1(ir_fea_patten)
        ir_fea_patten_M = self.sig(ir_fea_patten_conv)

        rgb_fea_patten_reshape = rgb_fea_patten_M.view(-1, h_pool * w_pool)
        ir_fea_patte_reshape = ir_fea_patten_M.view(-1, h_pool * w_pool)

        # concatenated_fea_patteen = torch.cat((rgb_fea_patten_reshape, ir_fea_patte_reshape), dim=0) # 只选RGB和IR本身的特征
        # concatenated_fea_patteen = torch.cat((rgb_fea_patten_reshape, ir_fea_patte_reshape, rgb_fea_patten_low_reshape, rgb_fea_patten_high_reshape, ir_fea_patten_low_reshape, ir_fea_patten_high_reshape), dim=0) # 选RGB和IR本身的特征，且选RGB和IR提取出的低频及高频特征
        len_fea_half = len(rgb_fea_patten_high_reshape) // 8  # 2 4 8
        concatenated_fea_patteen = torch.cat((rgb_fea_patten_reshape, ir_fea_patte_reshape,
                                             rgb_fea_patten_high_reshape[:len_fea_half], ir_fea_patten_high_reshape[:len_fea_half]), dim=0)  # 选RGB和IR本身的特征，且选RGB和IR提取出的高频特征
        # concatenated_fea_patteen = torch.cat((rgb_fea_patten_reshape, ir_fea_patte_reshape), dim=0) # 选RGB和IR本身的特征，且选RGB和IR提取出的高频特征

        # L_rgb = loss_function(rgb_fea_patten_reshape)
        # L_ir = loss_function(ir_fea_patte_reshape)
        self.pattenLoss = Seperation_loss(concatenated_fea_patteen)
        # print("self.pattenLoss:",self.pattenLoss)
        # self.pattenLoss =  L_rgb + L_ir
        # print("self.pattenLoss:{}".format(self.pattenLoss))

        rgb_fea_PT = self.conv2(rgb_fea_patten_M)
        ir_fea_PT = self.conv2(ir_fea_patten_M)

        P_rgb_fea = rgb_fea_PT * rgb_fea
        P_ir_fea = ir_fea_PT * ir_fea

        rgb_fea_flat = P_rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = P_ir_fea.view(bs, c, -1)  # flatten the feature
        ###########################################################

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # # pad token embeddings along number of tokens dimension
        # rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        # ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature

        # print("rgb_fea_flat.shape:{}".format(rgb_fea_flat.shape)) # rgb_fea_flat.shape:torch.Size([4, 128, 64])
        # print("ir_fea_flat.shape:{}".format(ir_fea_flat.shape)) # ir_fea_flat.shape:torch.Size([4, 128, 64])

        token_embeddings = torch.cat(
            [rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        # print("token_embeddings.shape:{}".format(token_embeddings.shape)) # token_embeddings.shape: torch.Size([4, 128, 128])

        token_embeddings = token_embeddings.permute(
            0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)
        # print("Permute token_embeddings.shape:{}".format(token_embeddings.shape)) #  Permute token_embeddings.shape: torch.Size([4, 128, 128])

        # transformer
        # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.drop(self.pos_emb + token_embeddings)
        # print("Pre Transfomer x.shape:{}".format(x.shape)) #  Pre Transfomer x.shape:torch.Size([4, 128, 128])

        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(
            bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(
            bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(
            rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out, self.pattenLoss


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(
            *[TransformerLayer(c2, num_heads) for _ in range(num_layers)])
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2)
        p = p.unsqueeze(0)
        p = p.transpose(0, 3)
        p = p.squeeze(3)
        e = self.linear(p)
        x = p + e

        x = self.tr(x)
        x = x.unsqueeze(3)
        x = x.transpose(0, 3)
        x = x.reshape(b, self.c2, w, h)
        return x


class Bottleneck(nn.Module):
    # Standard bottleneck
    # ch_in, ch_out, shortcut, groups, expansion
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super(C3, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(
            *[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            # suppress torch 1.9.0 max_pool2d() warning
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)


class AdaptiveModule3(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(AdaptiveModule3, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(in_channels)
        print("in_channels:", in_channels)
        print("out_channels:", out_channels)
        # Half_c = int(in_channels/2)
        # self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(in_channels)
        # self.relu1 = nn.LeakyReLU(0.1)
        # self.sobel = SobelConv2d(Half_c, Half_c)
        self.conv2 = nn.Conv2d(in_channels, in_channels * 8,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels * 8)
        self.relu2 = nn.LeakyReLU(0.1)

        # self.sobel = SobelConv2d(Half_c * 4, Half_c * 4)
        self.sobel = EnhanceConv2d(in_channels * 8, in_channels * 8)

        self.conv3 = nn.Conv2d(in_channels * 8, in_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.LeakyReLU(0.1)
        # self.adaptive_weight = nn.Parameter(torch.ones(out_channels), requires_grad=True)

    def forward(self, x):
        # print("x.shape:",x.shape)
        # x_2 = torch.cat((x, x), dim=1)
        # print("x_2.shape:", x_2.shape)

        # conv_1 = self.conv1(x)
        # bn_1 = self.bn1(conv_1)
        # relu_1 = self.relu1(bn_1)
        # out = self.sobel(out)

        conv_2 = self.conv2(x)
        bn_2 = self.bn2(conv_2)
        relu_2 = self.relu2(bn_2)
        sobel_1 = self.sobel(relu_2)
        # print("sobel Lauched! ")
        # concat_1 = torch.cat((relu_2, sobel_1), dim=1)
        # concat_2 = torch.cat((concat_1, x), dim=1)
        add_1 = relu_2 + sobel_1
        conv_3 = self.conv3(add_1)
        bn_3 = self.bn3(conv_3)
        out = self.relu3(bn_3)
        out = out + x
        # print("sobel launched!")
        # # out = out * self.adaptive_weight.view(1, -1, 1, 1)
        # print("AdaptiveModule Lauched! ")
        return out


class EnhanceConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, requires_grad=True):
        assert kernel_size % 2 == 1, 'EnhanceConv2d\'s kernel_size must be odd.'
        assert out_channels % 8 == 0, 'EnhanceConv2d\'s out_channels must be a multiple of 8.'
        assert out_channels % groups == 0, 'EnhanceConv2d\'s out_channels must be a multiple of groups.'

        super(EnhanceConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
        self.bias = bias if requires_grad else False

        if self.bias:
            self.bias = nn.Parameter(torch.zeros(
                size=(out_channels,), dtype=torch.float32), requires_grad=True)
        else:
            self.bias = None

        self.sobel_weight = nn.Parameter(torch.zeros(
            size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)

        # Initialize the EnhanceConv2d kernal
        kernel_mid = kernel_size // 2
        for idx in range(out_channels):
            if idx % 8 == 0:  # 水平
                self.sobel_weight[idx, :, 0, :] = -1
                self.sobel_weight[idx, :, 0, kernel_mid] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                self.sobel_weight[idx, :, -1, kernel_mid] = 2
            elif idx % 8 == 1:  # 垂直
                self.sobel_weight[idx, :, :, 0] = -1
                self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                self.sobel_weight[idx, :, kernel_mid, -1] = 2
            elif idx % 8 == 2:  # 左上右下对角线
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size -
                                      1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            elif idx % 8 == 3:  # 左下右上对角线
                self.sobel_weight[idx, :, 0, 0] = -2
                for i in range(0, kernel_mid + 1):
                    self.sobel_weight[idx, :, kernel_mid - i, i] = -1
                    self.sobel_weight[idx, :, kernel_size -
                                      1 - i, kernel_mid + i] = 1
                self.sobel_weight[idx, :, -1, -1] = 2
            elif idx % 8 == 4:  # 拉普拉斯算子
                self.sobel_weight[idx, :, 0, kernel_mid] = 1
                self.sobel_weight[idx, :, kernel_mid, :] = 1
                self.sobel_weight[idx, :, kernel_mid, kernel_mid] = -4
                self.sobel_weight[idx, :, -1, kernel_mid] = 1
            elif idx % 8 == 5:  # 负拉普拉斯算子
                self.sobel_weight[idx, :, 0, kernel_mid] = 1
                self.sobel_weight[idx, :, kernel_mid, :] = 1
                self.sobel_weight[idx, :, kernel_mid, kernel_mid] = 4
                self.sobel_weight[idx, :, -1, kernel_mid] = 1
            elif idx % 8 == 6:  # Prewitt算子（水平方向）
                self.sobel_weight[idx, :, 0, :] = -1
                # self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, -1, :] = 1
                # self.sobel_weight[idx, :, kernel_mid, -1] = 2
            else:              # Prewitt算子（垂直方向）
                self.sobel_weight[idx, :, :, 0] = -1
                # self.sobel_weight[idx, :, kernel_mid, 0] = -2
                self.sobel_weight[idx, :, :, -1] = 1
                # self.sobel_weight[idx, :, kernel_mid, -1] = 2

        # print("sobel weight:", self.sobel_weight)
        # Define the trainable sobel factor
        # self.sobel_factor = nn.Parameter( torch.FloatTensor(out_channels, 1, 1, 1, device="cuda") )
        if requires_grad:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=True)
        else:
            self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
                                             requires_grad=False)

    def forward(self, x):
        # if torch.cuda.is_available():
        #     self.sobel_factor = self.sobel_factor.cuda()
        #     if isinstance(self.bias, nn.Parameter):
        #         self.bias = self.bias.cuda()
        sobel_weight = self.sobel_weight
        sobel_weight = self.sobel_weight * self.sobel_factor

        # if torch.cuda.is_available():
        #     sobel_weight = sobel_weight.cuda()
        #     bias = self.bias.cuda()

        # print("x.shape:", x.shape)

        out = F.conv2d(x, sobel_weight, self.bias, self.stride,
                       self.padding, self.dilation, self.groups)
        # print("out.shape:", out.shape)
        return out


class Add(nn.Module):
    #  Add two tensors
    def __init__(self, arg):
        super(Add, self).__init__()
        self.arg = arg

    def forward(self, x):
        return torch.add(x[0], x[1])


class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])


class NMS(nn.Module):
    # Non-Maximum Suppression (NMS) module
    conf = 0.25  # confidence threshold
    iou = 0.45  # IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self):
        super(NMS, self).__init__()

    def forward(self, x):
        return non_max_suppression(x[0], conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)


class autoShape(nn.Module):
    # input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    classes = None  # (optional list) filter by class

    def __init__(self, model):
        super(autoShape, self).__init__()
        self.model = model.eval()

    def autoshape(self):
        # model already converted to model.autoshape()
        print('autoShape already enabled, skipping... ')
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   filename:   imgs = 'data/images/zidane.jpg'
        #   URI:             = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg')  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_synchronized()]
        p = next(self.model.parameters())  # for device and type
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=p.device.type != 'cpu'):
                # inference
                return self.model(imgs.to(p.device).type_as(p), augment, profile)

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (
            1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, str):  # filename or uri
                im, f = np.asarray(Image.open(requests.get(
                    im, stream=True).raw if im.startswith('http') else im)), im
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(im), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                # reverse dataloader .transpose(2, 0, 1)
                im = im.transpose((1, 2, 0))
            im = im[:, :, :3] if im.ndim == 3 else np.tile(
                im[:, :, None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(
                im)  # update
        shape1 = [make_divisible(x, int(self.stride.max()))
                  for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1, auto=False)[0]
             for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / \
            255.  # uint8 to fp16/32
        t.append(time_synchronized())

        with amp.autocast(enabled=p.device.type != 'cpu'):
            # Inference
            y = self.model(x, augment, profile)[0]  # forward
            t.append(time_synchronized())

            # Post-process
            y = non_max_suppression(
                y, conf_thres=self.conf, iou_thres=self.iou, classes=self.classes)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_synchronized())
            return Detections(imgs, y, files, t, self.names, x.shape)


class Detections:
    # detections class for YOLOv5 inference results
    def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
        super(Detections, self).__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*[im.shape[i] for i in [1, 0, 1, 0]], 1., 1.], device=d)
              for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 /
                       self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            str = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    # add to string
                    str += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "
                if show or save or render or crop:
                    for *box, conf, cls in pred:  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            save_one_box(
                                box, im, file=save_dir / 'crops' / self.names[int(cls)] / self.files[i])
                        else:  # all others
                            plot_one_box(box, im, label=label,
                                         color=colors(cls))

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(
                im, np.ndarray) else im  # from np
            if pprint:
                print(str.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                print(f"{'Saved' * (i == 0)} {f}", end=',' if i <
                      self.n - 1 else f' to {save_dir}\n')
            if render:
                self.imgs[i] = np.asarray(im)

    def print(self):
        self.display(pprint=True)  # print results
        print(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' % self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(
            save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save_dir='runs/hub/exp'):
        save_dir = increment_path(
            save_dir, exist_ok=save_dir != 'runs/hub/exp', mkdir=True)  # increment save_dir
        self.display(crop=True, save_dir=save_dir)  # crop results
        print(f'Saved results to {save_dir}\n')

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]]
                  for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s)
             for i in range(self.n)]
        for d in x:
            for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
                setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        super(Classify, self).__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (
            x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)


class SelfAttention(nn.Module):
    """
     Multi-head masked self-attention layer
    """

    def __init__(self, d_model, d_k, d_v, h, attn_pdrop=.1, resid_pdrop=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(SelfAttention, self).__init__()
        assert d_k % h == 0
        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        # key, query, value projections for all heads
        self.que_proj = nn.Linear(d_model, h * self.d_k)  # query projection
        self.key_proj = nn.Linear(d_model, h * self.d_k)  # key projection
        self.val_proj = nn.Linear(d_model, h * self.d_v)  # value projection
        self.out_proj = nn.Linear(h * self.d_v, d_model)  # output projection

        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, attention_mask=None, attention_weights=None):
        '''
        Computes Self-Attention
        Args:
            x (tensor): input (token) dim:(b_s, nx, c),
                b_s means batch size
                nx means length, for CNN, equals H*W, i.e. the length of feature maps
                c means channel, i.e. the channel of feature maps
            attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
            attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        Return:
            output (tensor): dim:(b_s, nx, c)
        '''

        b_s, nq = x.shape[:2]
        nk = x.shape[1]
        q = self.que_proj(x).view(b_s, nq, self.h, self.d_k).permute(
            0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.key_proj(x).view(b_s, nk, self.h, self.d_k).permute(
            0, 2, 3, 1)  # (b_s, h, d_k, nk) K^T
        v = self.val_proj(x).view(b_s, nk, self.h, self.d_v).permute(
            0, 2, 1, 3)  # (b_s, h, nk, d_v)

        # Self-Attention
        #  :math:`(\text(Attention(Q,K,V) = Softmax((Q*K^T)/\sqrt(d_k))`
        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)

        # weight and mask
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # get attention matrix
        att = torch.softmax(att, -1)
        att = self.attn_drop(att)

        # output
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(
            b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.resid_drop(self.out_proj(out))  # (b_s, nq, d_model)

        return out


class myTransformerBlock(nn.Module):
    """ Transformer block """

    def __init__(self, d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop):
        """
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        :param block_exp: Expansion factor for MLP (feed foreword network)

        """
        super().__init__()
        self.ln_input = nn.LayerNorm(d_model)
        self.ln_output = nn.LayerNorm(d_model)
        self.sa = SelfAttention(d_model, d_k, d_v, h, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, block_exp * d_model),
            # nn.SiLU(),  # changed from GELU
            nn.GELU(),  # changed from GELU
            nn.Linear(block_exp * d_model, d_model),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x):
        bs, nx, c = x.size()

        x = x + self.sa(self.ln_input(x))
        x = x + self.mlp(self.ln_output(x))

        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, d_model, h=8, block_exp=4,
                 n_layer=8, vert_anchors=8, horz_anchors=8,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        self.n_embd = d_model
        self.vert_anchors = vert_anchors
        self.horz_anchors = horz_anchors

        d_k = d_model
        d_v = d_model

        # positional embedding parameter (learnable), rgb_fea + ir_fea
        self.pos_emb = nn.Parameter(torch.zeros(
            1, 2 * vert_anchors * horz_anchors, self.n_embd))

        # transformer
        self.trans_blocks = nn.Sequential(*[myTransformerBlock(d_model, d_k, d_v, h, block_exp, attn_pdrop, resid_pdrop)
                                            for layer in range(n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(self.n_embd)

        # regularization
        self.drop = nn.Dropout(embd_pdrop)

        # avgpool
        self.avgpool = nn.AdaptiveAvgPool2d(
            (self.vert_anchors, self.horz_anchors))

        # init weights
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, h, w = rgb_fea.shape

        # -------------------------------------------------------------------------
        # AvgPooling
        # -------------------------------------------------------------------------
        # AvgPooling for reduce the dimension due to expensive computation
        rgb_fea = self.avgpool(rgb_fea)
        ir_fea = self.avgpool(ir_fea)

        # -------------------------------------------------------------------------
        # Transformer
        # -------------------------------------------------------------------------
        # pad token embeddings along number of tokens dimension
        rgb_fea_flat = rgb_fea.view(bs, c, -1)  # flatten the feature
        ir_fea_flat = ir_fea.view(bs, c, -1)  # flatten the feature
        token_embeddings = torch.cat(
            [rgb_fea_flat, ir_fea_flat], dim=2)  # concat
        token_embeddings = token_embeddings.permute(
            0, 2, 1).contiguous()  # dim:(B, 2*H*W, C)

        # transformer
        # sum positional embedding and token    dim:(B, 2*H*W, C)
        x = self.drop(self.pos_emb + token_embeddings)
        x = self.trans_blocks(x)  # dim:(B, 2*H*W, C)

        # decoder head
        x = self.ln_f(x)  # dim:(B, 2*H*W, C)
        x = x.view(bs, 2, self.vert_anchors, self.horz_anchors, self.n_embd)
        x = x.permute(0, 1, 4, 2, 3)  # dim:(B, 2, C, H, W)

        # 这样截取的方式, 是否采用映射的方式更加合理？
        rgb_fea_out = x[:, 0, :, :, :].contiguous().view(
            bs, self.n_embd, self.vert_anchors, self.horz_anchors)
        ir_fea_out = x[:, 1, :, :, :].contiguous().view(
            bs, self.n_embd, self.vert_anchors, self.horz_anchors)

        # -------------------------------------------------------------------------
        # Interpolate (or Upsample)
        # -------------------------------------------------------------------------
        rgb_fea_out = F.interpolate(
            rgb_fea_out, size=([h, w]), mode='bilinear')
        ir_fea_out = F.interpolate(ir_fea_out, size=([h, w]), mode='bilinear')

        return rgb_fea_out, ir_fea_out



class VAE(nn.Module):
    def __init__(self, input_channels, latent_dim=16, input_size=640):
        super(VAE, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # nn.AdaptiveAvgPool2d((640, 640)),
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(64 * (input_size // 4) *
                               (input_size // 4), latent_dim)
        self.fc_logvar = nn.Linear(
            64 * (input_size // 4) * (input_size // 4), latent_dim)

        # 解码器部分
        self.decoder_fc_output_shape = (64, input_size // 4, input_size // 4)
        self.decoder_fc = nn.Linear(
            latent_dim, 64 * (input_size // 4) * (input_size // 4))

        self.decoder = nn.Sequential(
            # 将线性层的输出重塑为四维张量
            nn.Unflatten(1, self.decoder_fc_output_shape),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_channels, kernel_size=3,
                               stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        z = self.decoder_fc(z)  # 先通过全连接层
        return self.decoder(z), mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


class RecContrastiveLoss(nn.Module):
    """
    对比损失类，鼓励正对接近而负对远离。
    """

    def __init__(self, margin=1.0):
        super(RecContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        计算对比损失。
        anchor - 当前模态的特征。
        positive - 同一实例的另一模态特征。
        negative - 不同实例的任一模态特征。
        """
        positive_distance = F.pairwise_distance(anchor, positive, 2)
        # negative_distance = F.pairwise_distance(anchor, negative, 2)
        # losses = torch.relu(positive_distance - negative_distance + self.margin)

        losses = torch.relu(positive_distance + self.margin)
        return losses.mean()
############################


class ModalitySpecificFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(ModalitySpecificFeatureExtractor, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 添加更多层根据需要...
        )

    def forward(self, x):
        x = self.layers(x)
        return x
############################


class ModalityAgnosticFeatureExtractor(nn.Module):
    def __init__(self, in_channels):
        super(ModalityAgnosticFeatureExtractor, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 添加更多层根据需要...
        )

    def forward(self, x):
        x = self.layers(x)
        return x
############################


class DecoderNetwork(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderNetwork, self).__init__()
        self.decoder = nn.Sequential(
            # 假设第一层将通道数从128减少到64
            nn.ConvTranspose2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 进一步减少通道数，并逐渐接近目标输出通道数
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            # 最后一层输出期望的通道数
            nn.ConvTranspose2d(
                32, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
            # 根据需要调整卷积核大小、步长和填充
        )

    def forward(self, x):
        x = self.decoder(x)
        return x


############################
def get_negative_features(batch_features, indices=None):
    """
    从批次中获取负样本特征。这个函数随机打乱当前批次的特征以生成负样本。
    batch_features: 当前批次的特征张量，形状为 (batch_size, feature_dim)。
    indices: 可选的，用于打乱的索引。
    """
    if indices is None:
        indices = torch.randperm(batch_features.size(0))
    negative_features = batch_features[indices]
    return negative_features


def reconstruction_loss(reconstructed, original):
    return nn.MSELoss()(reconstructed, original)


def vae_loss(recon_x, x, mu, logvar):
    # 假设 recon_x 已经通过sigmoid函数激活
    # 计算重建损失 (BCE)
    BCE = F.binary_cross_entropy_with_logits(recon_x, x, reduction='sum')

    # 计算KL散度
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
############################################################################
