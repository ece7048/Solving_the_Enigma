from __future__ import division, print_function
import numpy as np
import torch
from typing import Dict, Tuple

from monai import transforms
from SR_XAI.utilities.preprocessing import GraphImageDataset

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
    """Build train, validation, and test loaders for one NIfTI image family."""
    portion=0.3
    portion2=0.2
    train_transform = transforms.Compose(
    [transforms.LoadImaged(keys=["image"],image_only=False, ensure_channel_first=True),
    transforms.Resized(keys=["image"],spatial_size=(sx,sy,sz)),
    transforms.EnsureTyped(keys="image", track_meta=False),
    ]
    )
    val_transform = transforms.Compose(
    [transforms.LoadImaged(keys=["image"],image_only=False, ensure_channel_first=True),
     transforms.Resized(keys=["image"],spatial_size=(sx,sy,sz)),
        transforms.EnsureTyped(keys="image", track_meta=False),
        ])

    multitask = GraphImageDataset(DATA_ROOT1,DATA_ROOT2, train_transform, sx,sy,sz,class_base,cs,afs,name)
    whole=int(len(multitask))
    if whole <= 0:
        raise ValueError(f"No samples found for {name!r} in {DATA_ROOT1!r}")
    mid=int((len(multitask)*portion))
    mid2=mid+int((len(multitask)*portion2))
    if mid == 0 or mid2 <= mid or mid2 >= whole:
        raise ValueError(f"Dataset for {name!r} is too small to split: train={mid}, val={mid2-mid}, test={whole-mid2}")
    gen1 = torch.Generator().manual_seed(whole)
    whole_sampler = torch.utils.data.RandomSampler(multitask,num_samples=whole, generator=gen1)
    indices=list(whole_sampler)
    train_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:mid])
    trainloader = torch.utils.data.DataLoader(multitask, batch_size=batch_size,  sampler=train_sampler)
    print('SPLIT strategy : ',mid,mid2,whole)
    multitask2 = GraphImageDataset(DATA_ROOT1,DATA_ROOT2, val_transform, sx,sy,sz,class_base,cs,afs,name)
    gen2 = torch.Generator().manual_seed(whole)
    whole_sampler2 = torch.utils.data.RandomSampler(multitask2,num_samples=whole, generator=gen2)
    indices2=list(whole_sampler2)
    valid_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices2[mid:mid2])
    test_sampler=torch.utils.data.sampler.SubsetRandomSampler(indices2[mid2:])
    validloader= torch.utils.data.DataLoader(multitask2, batch_size=batch_size,  sampler=valid_sampler)
    testloader = torch.utils.data.DataLoader(multitask2, batch_size=batch_size,  sampler=test_sampler)
    return trainloader, validloader, testloader

def data_build(roi_size=16,bz=1,sx=112,sy=112,sz=112,class_base="1",DATA_ROOT1='None',DATA_ROOT2='None',cs='None',afs='None',name="seliency"):
    print("Centralized PyTorch training")
    print("Load data")
    trainloader, validloader, testloader= load_data(roi_size=roi_size,batch_size=bz,sx=sx,sy=sy,sz=sz,class_base=class_base,DATA_ROOT1=DATA_ROOT1,DATA_ROOT2=DATA_ROOT2,cs=cs,afs=afs,name=name)
        
    return trainloader, validloader, testloader
