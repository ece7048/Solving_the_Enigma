from __future__ import division, print_function
import torch.nn as nn
import torch.nn.functional as F
import math
import unfoldNd
import sys
import subprocess
import torch
from typing import Tuple, Dict
from torch import Tensor
import monai
from SR_XAI.utilities.preprocessing import *
from SR_XAI.models.SwiftUnet3D import *

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")



class XAI_SwinMT(torch.nn.Module):

    def __init__(self,sr_scale=2.0,dim=64,xai_num=10):
        super().__init__()
        self.swnet=SwinUNETR(img_size=(dim, dim, dim), in_channels=xai_num, out_channels=1, feature_size=24, drop_rate=0.0, attn_drop_rate=0.5, dropout_path_rate=0.0, use_checkpoint=True,)
        self.upsample=monai.networks.blocks.upsample.UpSample(spatial_dims=3,in_channels=1,out_channels=1,scale_factor=sr_scale)

    def forward(self,x,y):
        out1, enc3, enc2,enc1 =self.swnet(x)
        out = self.upsample(out1)
        return out, enc3


class Pos2Weight(nn.Module):
    def __init__(self, inC, kernel_size=3, outC=8, reduce=256):
        super(Pos2Weight, self).__init__()
        self.inC = inC
        self.kernel_size = kernel_size
        self.outC = outC
        self.meta_block = nn.Sequential(
            nn.Linear(4, reduce),
            nn.ReLU(inplace=True),
            nn.Linear(reduce, self.kernel_size * self.kernel_size * self.kernel_size * self.inC * self.outC)
        )

    def forward(self, x):
        x=torch.permute(x,(0,1))
        output = self.meta_block(x)
        return output


class MetaUpSampler(nn.Module):

    def __init__(self, n_feats, n_colors, kernel_size, reduce_step=256):
        super(MetaUpSampler, self).__init__()
        self.P2W = Pos2Weight(inC=n_feats, outC=n_colors, reduce=reduce_step)
        self.outC = n_colors
        self.inC = n_feats
        self.kernel_size = kernel_size
        self.unfold = unfoldNd.UnfoldNd(self.kernel_size,  dilation=1, padding=1, stride=1)

    def forward(self, lr_features, sr_scale):
        N, C, inH, inW, inD = lr_features.shape
        device = lr_features.device
        pos_mat, mask = input_matrix_wpn_new(inH, inW, inD, sr_scale)
        mask = mask.to(device)
        pos_mat = pos_mat.to(device)
        local_weight = self.P2W(pos_mat.view(pos_mat.size(2), -1))
        up_x = self.repeat_x(lr_features, sr_scale)
        cols=self.unfold(up_x)
        scale_int = math.ceil(sr_scale)
        local_weight = self.repeat_weight(local_weight, scale_int, inH, inW, inD)
        cols = cols.contiguous().view(
            cols.size(0) // (scale_int ** 3), scale_int ** 3, cols.size(1), cols.size(2), 1
        ).permute(0, 1, 3, 4, 2).contiguous()
        local_weight = local_weight.contiguous().view(
            inH, scale_int, inW, scale_int, inD, scale_int, -1, self.outC
        ).permute(1, 3, 5, 0, 2, 4, 6, 7).contiguous()
        local_weight = local_weight.contiguous().view(
            scale_int ** 3, inH *inW*inD, -1, self.outC
        )
        out = torch.matmul(cols, local_weight).permute(0, 1, 4, 2, 3)
        out = out.contiguous().view(
            N, scale_int, scale_int, scale_int, self.outC, inH, inW, inD
        ).permute(0, 4, 5, 1, 6, 2, 7, 3)
        out = out.contiguous().view(
            N, self.outC, scale_int * inH, scale_int * inW, scale_int * inD
        )

        out = torch.masked_select(out, mask)
        out = out.contiguous().view(
            N, self.outC, int(sr_scale * inH), int(sr_scale * inW), int(sr_scale * inD)
        )
        return out

    @staticmethod
    def repeat_x(x, scale):
        scale_int = math.ceil(scale)
        N, C, H, W ,D = x.size()
        x = x.view(N, C, H, 1, W, 1, D, 1)
        x = torch.cat([x] * scale_int, 3)
        x = torch.cat([x] * scale_int, 5)
        x = torch.cat([x] * scale_int, 7).permute(0, 3, 5, 7, 1, 2, 4, 6)
        x1=x.contiguous().view(-1, C, H, W, D)
        return x1

    @staticmethod
    def repeat_weight(weight, scale, inh, inw, ind):
        k = int(weight.size(0)**(1/3))
        if k%2!=0:
          k=k+1
        outw = inw * scale
        outh = inh * scale
        outd = ind * scale
        weight = weight.view(k, k, k, -1)
        scale_w = (outw + k - 1) // k
        scale_h = (outh + k - 1) // k
        scale_d = (outd + k - 1) // k
        weight = torch.cat([weight] * scale_h, 0)
        weight = torch.cat([weight] * scale_w, 1)
        weight = torch.cat([weight] * scale_d, 2)

        weight = weight[0:outh, 0:outw, 0:outd, :]

        return weight


def input_matrix_wpn_new(inH, inW, inD, scale, add_scale=True):
    '''
    inH, inW: the size of the feature maps
    scale: is the upsampling times
    '''
    outH, outW , outD= int(scale * inH), int(scale * inW), int(scale * inD)
    # Build interpolation offsets and a mask for valid high-resolution coordinates.
    scale_int = int(math.ceil(scale))
    h_offset = torch.ones(inH, scale_int, scale_int , 1)
    mask_h = torch.zeros(inH, scale_int, scale_int, 1)
    w_offset = torch.ones(1, inW, scale_int, scale_int)
    mask_w = torch.zeros(1, inW, scale_int, scale_int)
    d_offset = torch.ones(scale_int, 1, inD, scale_int)
    mask_d = torch.zeros(scale_int, 1, inD, scale_int)

    h_project_coord = torch.arange(0, outH, 1).mul(1.0 / scale)
    int_h_project_coord = torch.floor(h_project_coord)

    offset_h_coord = h_project_coord - int_h_project_coord
    int_h_project_coord = int_h_project_coord.int()

    w_project_coord = torch.arange(0, outW, 1).mul(1.0 / scale)
    int_w_project_coord = torch.floor(w_project_coord)

    offset_w_coord = w_project_coord - int_w_project_coord
    int_w_project_coord = int_w_project_coord.int()

    d_project_coord = torch.arange(0, outD, 1).mul(1.0 / scale)
    int_d_project_coord = torch.floor(d_project_coord)

    offset_d_coord = d_project_coord - int_d_project_coord
    int_d_project_coord = int_d_project_coord.int()

    flag = 0
    number = 0
    for i in range(outH):
        if int_h_project_coord[i] == number:
            h_offset[int_h_project_coord[i], flag, flag, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], flag, flag, 0] = 1
            flag += 1
        else:
            h_offset[int_h_project_coord[i], 0, 0, 0] = offset_h_coord[i]
            mask_h[int_h_project_coord[i], 0, 0, 0] = 1
            number += 1
            flag = 1
    flag = 0
    number = 0
    for i in range(outW):
        if int_w_project_coord[i] == number:
            w_offset[0,int_w_project_coord[i], flag, flag] = offset_w_coord[i]
            mask_w[0,int_w_project_coord[i], flag, flag] = 1
            flag += 1
        else:
            w_offset[0, int_w_project_coord[i], 0, 0] = offset_w_coord[i]
            mask_w[0, int_w_project_coord[i], 0, 0] = 1
            number += 1
            flag = 1

    flag = 0
    number = 0
    for i in range(outD):
        if int_d_project_coord[i] == number:
            d_offset[flag, 0, int_d_project_coord[i], flag] = offset_d_coord[i]
            mask_d[flag, 0, int_d_project_coord[i], flag] = 1
            flag += 1
        else:
            d_offset[0,0,int_d_project_coord[i], 0] = offset_d_coord[i]
            mask_d[0,0,int_d_project_coord[i], 0] = 1
            number += 1
            flag = 1

    h_offset_coord = torch.cat([h_offset] * (scale_int * inW)* (scale_int *inH), 3).view(-1,  scale_int * inW, scale_int * inH, 1)
    w_offset_coord = torch.cat([w_offset] * (scale_int * inH)* (scale_int *inD), 0).view(-1,  scale_int * inH, scale_int * inD, 1)
    d_offset_coord = torch.cat([d_offset] * (scale_int * inD)* (scale_int *inW), 1).view(-1,  scale_int * inD, scale_int * inW, 1)
    mask_h = torch.cat([mask_h] * (scale_int * inW)* (scale_int *inH)* (scale_int*scale_int), 3).view(-1,  scale_int * inW, scale_int * inH, 1)
    mask_w = torch.cat([mask_w] * (scale_int * inH)* (scale_int *inD)* (scale_int*scale_int), 0).view(-1,  scale_int * inH, scale_int * inD, 1)
    mask_d = torch.cat([mask_d] * (scale_int * inD)* (scale_int *inW)* (scale_int*scale_int), 1).view(-1,  scale_int * inD, scale_int * inW, 1)

    pos_mat = torch.cat((h_offset_coord, w_offset_coord, d_offset_coord), 3)
    mask_mat = torch.sum(torch.cat((mask_h, mask_w, mask_d), 3), 3).view(-1, scale_int * inH, scale_int * inW, scale_int * inD )
    mask_mat = mask_mat.eq(3)

    i = 1
    h, w, d , _= pos_mat.size()
    while ((pos_mat[i][0][0][0] >= 1e-6) and (i < (h-1))):
        i = i + 1

    j = 1
    h, w, d , _= pos_mat.size()
    while (pos_mat[0][j][0][1] >= 1e-6 and j < (w-1)):
        j = j + 1

    k = 1
    h, w, d , _= pos_mat.size()
    while (pos_mat[0][0][k][2] >= 1e-6 and k < (d-1)):
        k = k + 1

    pos_mat_small = pos_mat[0:i, 0:j, 0:k, :]
    pos_mat_small = pos_mat_small.contiguous().view(1, 1, -1, 3)

    if add_scale:
        scale_mat = torch.zeros(1,1, 1)
        scale_mat[0,0,0] = 1.0 / scale
        scale_mat = torch.cat([scale_mat] * (pos_mat_small.size(2)), 0)
        pos_mat_small = torch.cat((scale_mat.view(1,1,-1,1), pos_mat_small), 3)

    return pos_mat_small, mask_mat

class MeanShift(nn.Conv3d):

    def __init__(self, mean=(0.,), std=(1.0,), mode='sub'):
        if len(mean) != len(std):
            raise ValueError('Size of means and stds should be the same')
        nc = len(mean)
        super(MeanShift, self).__init__(nc, nc, kernel_size=1)
        std = torch.Tensor(std)

        if mode == 'sub':
            self.weight.data = torch.eye(nc).view(nc, nc, 1, 1, 1) / std.view(nc, 1, 1, 1, 1)
            self.bias.data = -1 * torch.Tensor(mean) / std
        elif mode == 'add':
            self.weight.data = torch.eye(nc).view(nc, nc, 1, 1, 1) * std.view(nc, 1, 1, 1, 1)
            self.bias.data = 1 * torch.Tensor(mean)
        for p in self.parameters():
            p.requires_grad = False


import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange,repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_3d(patches, temperature = 10000, dtype = torch.float32):
    _, f, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    z, y, x = torch.meshgrid(
        torch.arange(f, device = device),
        torch.arange(h, device = device),
        torch.arange(w, device = device),
    indexing = 'ij')

    fourier_dim = dim // 6

    omega = torch.arange(fourier_dim, device = device) / (fourier_dim - 1)
    omega = 1. / (temperature ** omega)

    z = z.flatten()[:, None] * omega[None, :]
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 

    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos(), z.sin(), z.cos()), dim = 1)

    pe = F.pad(pe, (0, dim - (fourier_dim * 6))) # pad if feature dimension not cleanly divisible by 6
    return pe.type(dtype)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class multi_ViT3D(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by the frame patch size'

        num_patches = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim = channels * patch_height * patch_width * frame_patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b f h w (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(int(dim), depth, heads, int(2*dim_head), mlp_dim)
        self.dim=dim
        self.to_latent = nn.Identity()
        self.linear_head_pre = nn.Linear(2*dim, dim)
        self.linear_head = nn.Linear(dim, num_classes)

    def forward(self, base,image):
        *_, h, w, dtype = *base.shape, base.dtype

        x = self.to_patch_embedding(base)
        y = self.to_patch_embedding(image)
        pex = posemb_sincos_3d(x)
        pey = posemb_sincos_3d(y)
        x = rearrange(x, 'b ... d -> b (...) d') + pex
        y = rearrange(y, 'b ... d -> b (...) d') + pey
        z=torch.cat((x,y),dim=1)
        z=self.transformer(z)
        z = z.mean(dim = 1)
        z = self.to_latent(z)
        z=z.view(-1,int(self.dim**0.5),int(self.dim**0.5))
        return z

class XAISR(nn.Module):

    def __init__(self,sr_scale=2.0,dim=64,xai_num=10) -> None:
        super(XAISR, self).__init__()
        self.scale=sr_scale
        self.dim=dim
        self.vit=multi_ViT3D(image_size=dim, image_patch_size=int(dim/4), frames=dim, frame_patch_size=int(dim/4), num_classes=1, dim=dim*dim, depth=2, heads=8, mlp_dim=int(2*dim), channels = 1, dim_head = dim)
        self.upsample=monai.networks.blocks.upsample.UpSample(spatial_dims=3,in_channels=1,out_channels=1,scale_factor=sr_scale)
        self.sg=torch.nn.Sigmoid()
        self.passerup=nn.Conv3d((1),(1),kernel_size=1)
        self.batch_norm=nn.BatchNorm1d(xai_num)
        self.batch_norm2=nn.BatchNorm2d(xai_num)

    def forward(self,x: Tensor, y: Tensor) -> Tensor:
        base_num=x.shape[1]
        wap=[]
        max_value=0
        for i in range(base_num):
            base=x[:,i]
            base_ex=torch.unsqueeze(base,dim=1)
            if torch.all(base==0):
                ws=torch.zeros((x.shape[0],1,self.dim,self.dim),device=device)
            else:
                w=self.vit(base_ex,y)
                ws=torch.unsqueeze(w,dim=1)
              
            wap.append(ws)
        weight=torch.cat(wap,dim=1)
        if weight.shape[0]==1:
            weight=torch.squeeze(weight)
            weight=self.sg(weight+0.02) # be sure it is not very small 
            weight=torch.unsqueeze(weight,dim=0)
        else:
            weight=self.batch_norm2(weight)
        z=weight_mean(x,weight,axis=1,batch='on')
        xtot=torch.reshape(z,(-1,1,int(self.dim),int(self.dim),int(self.dim)))
        xcon=self.passerup(xtot)  
        outmax = self.upsample(xcon)
        return outmax, weight





# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Any, Union, Optional

import torch
import torch.nn as nn

from monai.networks.layers.factories import Conv, Norm, Pool
from monai.networks.layers.utils import get_pool_layer
from monai.utils import ensure_tuple_rep
from monai.utils.module import look_up_option



def get_inplanes():
    return [64, 128, 256, 512]


def get_avgpool():
    return [0, 1, (1, 1), (1, 1, 1)]


class ResNetBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        inplace: bool = True,
        downsample: Union[nn.Module,partial] = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels.
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for first conv layer.
            downsample: which downsample layer to use.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=3, padding=1, stride=stride, bias=False)
        self.bn1 = norm_type(planes)
        self.relu = nn.ReLU(inplace=inplace)
        self.conv2 = conv_type(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.ReLU(inplace=inplace)



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNetBottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        spatial_dims: int = 3,
        stride: int = 1,
        inplace:bool=True,
        downsample: Union[nn.Module,partial] = None,
    ) -> None:
        """
        Args:
            in_planes: number of input channels.
            planes: number of output channels (taking expansion into account).
            spatial_dims: number of spatial dimensions of the input image.
            stride: stride to use for second conv layer.
            downsample: which downsample layer to use.
        """

        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        self.conv1 = conv_type(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_type(planes)
        self.conv2 = conv_type(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_type(planes)
        self.conv3 = conv_type(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_type(planes * self.expansion)
        self.relu = nn.ReLU(inplace=inplace)
        self.downsample = downsample
        self.stride = stride

        self.relu2 = nn.ReLU(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out




class ResNet(nn.Module):
    """
    ResNet based on: `Deep Residual Learning for Image Recognition <https://arxiv.org/pdf/1512.03385.pdf>`_
    and `Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet? <https://arxiv.org/pdf/1711.09577.pdf>`_.
    Adapted from `<https://github.com/kenshohara/3D-ResNets-PyTorch/tree/master/models>`_.

    Args:
        block: which ResNet block to use, either Basic or Bottleneck.
            ResNet block class or str.
            for Basic: ResNetBlock or 'basic'
            for Bottleneck: ResNetBottleneck or 'bottleneck'
        layers: how many layers to use.
        block_inplanes: determine the size of planes at each step. Also tunable with widen_factor.
        spatial_dims: number of spatial dimensions of the input image.
        n_input_channels: number of input channels for first convolutional layer.
        conv1_t_size: size of first convolution layer, determines kernel and padding.
        conv1_t_stride: stride of first convolution layer.
        no_max_pool: bool argument to determine if to use maxpool layer.
        shortcut_type: which downsample block to use. Options are 'A', 'B', default to 'B'.
            - 'A': using `self._downsample_basic_block`.
            - 'B': kernel_size 1 conv + norm.
        widen_factor: widen output for each layer.
        num_classes: number of output (classifications).
        feed_forward: whether to add the FC layer for the output, default to `True`.
        bias_downsample: whether to use bias term in the downsampling block when `shortcut_type` is 'B', default to `True`.

    """

    def __init__(
        self,
        block: Union[type[Union[ResNetBlock ,ResNetBottleneck]] ,str],
        layers: list[int],
        block_inplanes: list[int],
        spatial_dims: int = 3,
        n_input_channels: int = 3,
        conv1_t_size: Union[tuple[int] , int] = 7,
        conv1_t_stride: Union[tuple[int] , int] = 1,
        no_max_pool: bool = False,
        shortcut_type: str = "B",
        widen_factor: float = 1.0,
        num_classes: int = 400,
        feed_forward: bool = True,
        inplace: bool=True,
        bias_downsample: bool = True,  # for backwards compatibility (also see PR #5477)
    ) -> None:
        super().__init__()

        if isinstance(block, str):
            if block == "basic":
                block = ResNetBlock
            elif block == "bottleneck":
                block = ResNetBottleneck
            else:
                raise ValueError("Unknown block '%s', use basic or bottleneck" % block)

        conv_type: type[Union[nn.Conv1d , nn.Conv2d , nn.Conv3d]] = Conv[Conv.CONV, spatial_dims]
        norm_type: type[Union[nn.BatchNorm1d , nn.BatchNorm2d , nn.BatchNorm3d]] = Norm[Norm.BATCH, spatial_dims]
        pool_type: type[Union[nn.MaxPool1d , nn.MaxPool2d , nn.MaxPool3d]] = Pool[Pool.MAX, spatial_dims]
        avgp_type: type[Union[nn.AdaptiveAvgPool1d , nn.AdaptiveAvgPool2d , nn.AdaptiveAvgPool3d]] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims
        ]

        block_avgpool = get_avgpool()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.inplace=inplace
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool
        self.bias_downsample = bias_downsample

        conv1_kernel_size = ensure_tuple_rep(conv1_t_size, spatial_dims)
        conv1_stride = ensure_tuple_rep(conv1_t_stride, spatial_dims)

        self.conv1 = conv_type(
            n_input_channels,
            self.in_planes,
            kernel_size=conv1_kernel_size,  # type: ignore
            stride=conv1_stride,  # type: ignore
            padding=tuple(k // 2 for k in conv1_kernel_size),  # type: ignore
            bias=False,
        )
        self.bn1 = norm_type(self.in_planes)
        self.relu = nn.ReLU(inplace=inplace)
        self.maxpool = pool_type(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0], spatial_dims, shortcut_type)
        self.layer2 = self._make_layer(block, block_inplanes[1], layers[1], spatial_dims, shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, block_inplanes[2], layers[2], spatial_dims, shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, block_inplanes[3], layers[3], spatial_dims, shortcut_type, stride=2)
        self.avgpool = avgp_type(block_avgpool[spatial_dims])
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, num_classes) if feed_forward else None
        self.relu2 = nn.ReLU(inplace=inplace)

        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight), mode="fan_out", nonlinearity="relu")
            elif isinstance(m, norm_type):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(torch.as_tensor(m.bias), 0)

    def _downsample_basic_block(self, x: torch.Tensor, planes: int, stride: int, spatial_dims: int = 3) -> torch.Tensor:
        out: torch.Tensor = get_pool_layer(("avg", {"kernel_size": 1, "stride": stride}), spatial_dims=spatial_dims)(x)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), *out.shape[2:], dtype=out.dtype, device=out.device)
        out = torch.cat([out.data, zero_pads], dim=1)
        return out

    def _make_layer(
        self,
        block: type[Union[ResNetBlock , ResNetBottleneck]],
        planes: int,
        blocks: int,
        spatial_dims: int,
        shortcut_type: str,
        stride: int = 1,
    ) -> nn.Sequential:
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        norm_type: Callable = Norm[Norm.BATCH, spatial_dims]

        downsample: Union(nn.Module , partial) = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if look_up_option(shortcut_type, {"A", "B"}) == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    spatial_dims=spatial_dims,
                )
            else:
                downsample = nn.Sequential(
                    conv_type(
                        self.in_planes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=self.bias_downsample,
                    ),
                    norm_type(planes * block.expansion),
                )


        layers = [block(in_planes=self.in_planes, planes=planes, spatial_dims=spatial_dims, stride=stride, inplace=self.inplace, downsample=downsample)]
        self.in_planes = planes * block.expansion
        for _i in range(0, blocks):
            layers.append(block(self.in_planes, planes, spatial_dims=spatial_dims,inplace=self.inplace))

        return nn.Sequential(*layers)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        if self.fc is not None:
            x = self.fc(x)

        return x


def _resnet(
    arch: str,
    block: type[Union[ResNetBlock , ResNetBottleneck]],
    layers: list[int],
    block_inplanes: list[int],
    pretrained: bool,
    progress: bool,
    inplace=True,
    **kwargs: Any,
) -> ResNet:
    model: ResNet = ResNet(block, layers, block_inplanes, inplace=inplace, **kwargs)
    if pretrained:
        # Author of paper zipped the state_dict on googledrive,
        # so would need to download, unzip and read (2.8gb file for a ~150mb state dict).
        # Would like to load dict from url but need somewhere to save the state dicts.
        raise NotImplementedError(
            "Currently not implemented. You need to manually download weights provided by the paper's author"
            " and load then to the model with `state_dict`. See https://github.com/Tencent/MedicalNet"
            "Please ensure you pass the appropriate `shortcut_type` and `bias_downsample` args. as specified"
            "here: https://github.com/Tencent/MedicalNet/tree/18c8bb6cd564eb1b964bffef1f4c2283f1ae6e7b#update20190730"
        )
    return model


def resnet10(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-10 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes(), pretrained, progress, **kwargs)


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-18 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", ResNetBlock, [2, 2, 2, 2], get_inplanes(), pretrained, progress, **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, inplace=True, **kwargs: Any) -> ResNet:
    """ResNet-34 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", ResNetBlock, [3, 4, 6, 3], get_inplanes(), pretrained, progress, inplace=inplace, **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, inplace=True,  **kwargs: Any) -> ResNet:
    """ResNet-50 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 23 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", ResNetBottleneck, [3, 4, 6, 3], get_inplanes(), pretrained, progress, inplace=inplace, **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-101 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", ResNetBottleneck, [3, 4, 23, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-152 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", ResNetBottleneck, [3, 8, 36, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet200(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    """ResNet-200 with optional pretrained support when `spatial_dims` is 3.

    Pretraining from `Med3D: Transfer Learning for 3D Medical Image Analysis <https://arxiv.org/pdf/1904.00625.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on 8 medical datasets
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet200", ResNetBottleneck, [3, 24, 36, 3], get_inplanes(), pretrained, progress, **kwargs)




class networks:
    def __init__(self,name,device,inplace=True):
        self.net=name
        self.device=device
        self.inplace=inplace

    def build(self,c=2):
        if self.net=='resnet50':
            net=resnet50(spatial_dims=3,n_input_channels=1,num_classes=2,inplace=self.inplace).to(self.device)
        elif self.net=='resnet34':
            net=resnet34(spatial_dims=3,n_input_channels=1,num_classes=2,inplace=self.inplace).to(self.device)
        elif self.net=='resnet18':
            net=resnet18(spatial_dims=3,n_input_channels=1,num_classes=2).to(self.device)
        elif self.net=='efficient':
            net=monai.networks.nets.EfficientNetBN(model_name='efficientnet-l2',spatial_dims=3, in_channels=1,num_classes=c).to(self.device)
        elif self.net=='densenet121':
            net=monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=c).to(self.device)
        elif self.net=='seresnext50':
            net=monai.networks.nets.SEResNext50(spatial_dims=3, in_channels=1, out_channels=c).to(self.device)
        elif self.net=='highresnet50':
            net=monai.networks.nets.HighResNet(spatial_dims=3, in_channels=1, out_channels=c).to(self.device)
        else:
            print('nodefine network!!')

        return net
                                                                                                      
