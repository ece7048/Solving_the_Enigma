#Author: Michail Mamalakis
#Version: 0.1
#Licence:MIT
#email:mm2703@cam.ac.uk

from __future__ import division, print_function
import os
import torch
from SR_XAI import preprocessing
from SR_XAI.preprocessing import *
from torch.utils.data import Dataset, DataLoader
from torchdata.datapipes.iter import IterableWrapper
import nibabel as nib
import numpy as np
from scipy import ndimage
import skimage.transform as skTrans
#import torchvision.transforms as transforms
from monai.data import decollate_batch

from monai.transforms import (
    EnsureChannelFirstd,
    AsDiscrete,
    Compose,
    CropForegroundd,
    EnsureTyped,
    FgBgToIndicesd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    RandFlip,
    RandRotate,
    RandZoom,
    RandAffineGrid,
    RandGaussianNoise,
    RandShiftIntensity,
    NormalizeIntensity,
)
from monai import transforms
from monai.transforms import (
    AsDiscrete,
    Activations,
)
from torch import Tensor
from typing import Tuple, Dict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)

def load_data(roi_size=16,batch_size=1,sx=112,sy=112,sz=112,class_base="1",DATA_ROOT1='None',DATA_ROOT2='None',cs='None',afs='None',name='saliency') -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict]:
    """Load CIFAR-10 (training and test set)."""
    portion=0.3
    portion2=0.2
    r=roi_size
    train_transform = transforms.Compose(
    [transforms.LoadImaged(keys=["image"],image_only=False, ensure_channel_first=True),
    transforms.Resized(keys=["image"],spatial_size=(sx,sy,sz)),
    #write a Lmbda to take the labels check if have 2 and give class 0 or 1 respectivelry!! TODO!!
    #transforms.Lambdad(keys='label', func=lambda x: 0 if int(x.max())==0 else x+1, overwrite='class'),
    #transforms.RandSpatialCropSamplesd(
    #            keys=["image"],
    #            num_samples=16,
    #            roi_size=[roi_size,int(roi_size/2),int(roi_size/8)],
    #            random_size=False,random_center=False
    #        ),
    transforms.EnsureTyped(keys="image", track_meta=False),
    #transforms.RandFlipd( keys="image", prob=0.2, spatial_axis=1),
    #transforms.RandFlipd( keys="image", prob=0.1, spatial_axis=2),
    #transforms.RandFlipd( keys="image", prob=0.3, spatial_axis=0),
   # transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    #transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
    #transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.2),
    ]
    )
    val_transform = transforms.Compose(
    [transforms.LoadImaged(keys=["image"],image_only=False, ensure_channel_first=True),
     transforms.Resized(keys=["image"],spatial_size=(sx,sy,sz)),
        #transforms.SpatialCropd(keys=["image"],roi_center=(cx,cy,cz), roi_size=(sx,sy,sz)),
        #transforms.Lambdad(keys='label', func=lambda x: 0 if int(x.max())==0 else (x+1 if ...), overwrite='class'),
        #transforms.RandSpatialCropSamplesd(keys=["image", "label"],num_samples=8,roi_size=[64,64,64],random_size=False,random_center=False),
        transforms.EnsureTyped(keys="image", track_meta=False),
      #  transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        ])

    multitask = GraphImageDataset(DATA_ROOT1,DATA_ROOT2, train_transform, sx,sy,sz,class_base,cs,afs,name)
    whole=int(len(multitask))
    mid=int((len(multitask)*portion))
    mid2=mid+int((len(multitask)*portion2))
    gen1 = torch.Generator().manual_seed(whole)
    whole_sampler = torch.utils.data.RandomSampler(multitask,num_samples=whole, generator=gen1)
    indices=list(whole_sampler)
    train_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:mid])
    trainloader = torch.utils.data.DataLoader(multitask, batch_size=batch_size,  sampler=train_sampler) # ,num_workers=4)
    print('SPLIT strategy : ',mid,mid2,whole)
    multitask2 = GraphImageDataset(DATA_ROOT1,DATA_ROOT2, val_transform, sx,sy,sz,class_base,cs,afs,name)
    whole_sampler2 = torch.utils.data.RandomSampler(multitask2,num_samples=whole, generator=gen1)
    indices2=list(whole_sampler2)
    valid_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices2[mid:mid2])
    test_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices2[mid2:])
    validloader= torch.utils.data.DataLoader(multitask2, batch_size=batch_size,  sampler=valid_sampler)  #,num_workers=4)
    testloader = torch.utils.data.DataLoader(multitask2, batch_size=batch_size,  sampler=test_sampler)  #,num_workers=4)
    return trainloader, validloader, testloader

def data_build(roi_size=16,bz=1,sx=112,sy=112,sz=112,class_base="1",DATA_ROOT1='None',DATA_ROOT2='None',cs='None',afs='None',name="seliency"):
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, validloader, testloader= load_data(roi_size=roi_size,batch_size=bz,sx=sx,sy=sy,sz=sz,class_base=class_base,DATA_ROOT1=DATA_ROOT1,DATA_ROOT2=DATA_ROOT2,cs=cs,afs=afs,name=name)
        
    return trainloader, validloader, testloader
