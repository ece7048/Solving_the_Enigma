from __future__ import division, print_function

import nibabel as nib
from scipy import ndimage
import csv
import numpy as np
import os
import torch
import random

from torch.utils.data import Dataset, DataLoader
from torchdata.datapipes.iter import IterableWrapper
import nibabel as nib
import numpy as np
from scipy import ndimage
import skimage.transform as skTrans
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
)

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")

def next_or_restart(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator

def normalize_tensor(volume, out_range=255.0, eps=1e-8):
    min_value = torch.min(volume)
    max_value = torch.max(volume)
    denom = torch.clamp(max_value - min_value, min=eps)
    return ((volume - min_value) / denom) * out_range

def safe_ratio(num, den, eps=1e-8):
    return abs(num) / max(abs(den), eps)

def excel_label(filenamep='example.csv', given_name='908', cell_col=2, list_name=['absent','prominent','present']):
    with open(filenamep, 'r') as csvfile:
        reader = csv.reader(csvfile)
        cell_value = None
        for row in reader:
            if row[0] == given_name[:-3]: #_T1 we do not need in this case !!
                found_name = True
                cell_value = row[cell_col-1]  # adjust for 0-based indexing
                break
            else:
                found_name=False
    if cell_value==list_name[0]:
        cell_value=0
    elif cell_value==list_name[1]:
        cell_value=1
    elif cell_value==list_name[2]:
        cell_value=1 #(binary)
    else:
        cell_value=cell_value

    csvfile.close()
    return cell_value


def normalize(volume):
    min=np.min(volume)
    max=np.max(volume)
    volume[volume<min]=min
    volume[volume>max]=max
    denom=max-min
    if denom == 0:
        volume=np.zeros_like(volume)
    else:
        volume=(volume-min)/denom
    volume=volume.astype("float32")
    return volume

def resize_volume(img,d=112,w=112,h=112):
    """Resize across z-axis"""
    desired_depth = d
    desired_width = w
    desired_height = h
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    if len(img.shape)==4:
        if img.shape[1]==img.shape[2]:
            img=np.transpose(img, (1, 2, 0, 3))
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor,1), order=1)
    else:
        if img.shape[1]==img.shape[2]:
            img=np.transpose(img, (1, 2, 0))
        img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def calculate_weights(faith, complexity,l1,l2):

    if len(faith) != len(complexity):
        raise ValueError("Both lists must be of the same length.")
    max_faith = max(max(abs(f) for f in faith), 1e-8)
    max_complexity = max(max(abs(c) for c in complexity), 1e-8)
    normalized_faith = [abs(f) / max_faith for f in faith]
    normalized_complexity = [max(abs(c) / max_complexity, 1e-8) for c in complexity]

    weights = [l1 * f + l2 * (1 / c) for f, c in zip(normalized_faith, normalized_complexity)]
    return weights

def weight_initializer(ex,loader,net):
    from SR_XAI.utilities import utils

    loader_im=iter(loader)
    exp_iterator0=iter(ex[0])
    exp_iterator1=iter(ex[1])
    exp_iterator2=iter(ex[2])
    exp_iterator3=iter(ex[3])
    exp_iterator4=iter(ex[4])
    exp_iterator5=iter(ex[5])
    exp_iterator6=iter(ex[6])
    exp_iterator7=iter(ex[7])
    exp_iterator8=iter(ex[8]) 
    data1, exp_iterator0 = next_or_restart(exp_iterator0, ex[0])
    data2, exp_iterator1 = next_or_restart(exp_iterator1, ex[1])
    data3, exp_iterator2 = next_or_restart(exp_iterator2, ex[2])
    data4, exp_iterator3 = next_or_restart(exp_iterator3, ex[3])
    data5, exp_iterator4 = next_or_restart(exp_iterator4, ex[4])
    data6, exp_iterator5 = next_or_restart(exp_iterator5, ex[5])
    data7, exp_iterator6 = next_or_restart(exp_iterator6, ex[6])
    data8, exp_iterator7 = next_or_restart(exp_iterator7, ex[7])
    data9, exp_iterator8 = next_or_restart(exp_iterator8, ex[8])
    data, loader_im = next_or_restart(loader_im, loader)

    xai_images=torch.cat((data1["image"].to(device=device, dtype=torch.float),data2["image"].to(device=device, dtype=torch.float),data3["image"].to(device=device, dtype=torch.float),data4["image"].to(device=device, dtype=torch.float),data5["image"].to(device=device, dtype=torch.float),data6["image"].to(device=device, dtype=torch.float),data7["image"].to(device=device, dtype=torch.float),data8["image"].to(device=device, dtype=torch.float),data9["image"].to(device=device, dtype=torch.float)),1)
    imagesv=data["image"].to(device=device,dtype=torch.float)
    y_batchv=data["class"].to(device=device,dtype=torch.int)

    complexity=[]
    faith=[]
    rdn=random.randint(0,imagesv.shape[0]-1)
    for o in range(len(ex)):
        xai=xai_images[rdn,o]
        xai_ex=torch.unsqueeze(xai,dim=0)
        if torch.all(xai_ex==0):
            max_faith=0
            min_com=1000
        else:
            min_com,max_faith=utils.xai_score(imagesv[rdn],y_batchv[rdn],xai_ex,net=net,device=device)
        complexity.append(min_com)
        faith.append(max_faith)
        
    l1=0.6
    l2=0.4
    weights=calculate_weights(faith, complexity, 0.6, 0.4)
    return weights


def weight_mean(A,weight,axis=1,batch='off'):
    if torch.is_tensor(weight) and batch=='off':
        W=(weight)
        Wexp=W.view(1,W.shape[0],1,1,1)
        Z = torch.mul(A.to(device=device, dtype=torch.float), Wexp.to(device=device, dtype=torch.float))
        denom=torch.clamp(torch.sum(Wexp.to(device=device, dtype=torch.float)).to(device=device, dtype=torch.float), min=1e-8)
        out=torch.sum(Z, dim=axis) / denom
    else:
        W=(weight)
        if len(W.shape)==3:
            Wexp=W.view(-1,W.shape[1],W.shape[2],1,1)
        elif len(W.shape)==4:
            Wexp=W.view(-1,W.shape[1],W.shape[2],W.shape[3],1)
        elif len(W.shape)==5:
            Wexp=W.view(-1,W.shape[1],W.shape[2],W.shape[3],W.shape[4])
        else:
            Wexp=W.view(-1,W.shape[1],1,1,1)
        Z = torch.mul(A.to(device=device, dtype=torch.float), Wexp.to(device=device, dtype=torch.float))
        denom=torch.clamp(torch.sum(Wexp.to(device=device, dtype=torch.float)).to(device=device, dtype=torch.float), min=1e-8)
        out=torch.sum(Z, dim=axis) / denom
    
    return out
   
class GraphImageDataset(Dataset):
    def __init__(self, data_path:str = "", data_path2:str = "",transform=None,sx=112,sy=112,sz=112,class_base="1",cs='None',afs='None',name="saliency"):
        self.case_file=cs
        self.after_sample=afs
        self.data_path = data_path
        self.transform = transform
        self.inp=0
        self.sx,self.sy,self.sz=sx,sy,sz
        self.class_base=class_base
        self.sub=[]
        class_dir = os.path.join(data_path2, str(class_base))
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Class directory not found: {class_dir}")
        items=os.listdir(class_dir)
        items=sorted(items)
        self.name=name
        stop_at_case = cs not in (None, "", "None")
        for item in items:
            index=item.find("i")
            sub_itter=item[:index] if index > 0 else os.path.splitext(item)[0]
            if stop_at_case and str(sub_itter)==str(cs):
                break
            self.sub.append(sub_itter)
        if not self.sub:
            raise ValueError(f"No subjects found in {class_dir} before cs={cs!r}")

    def __len__(self):
        return len(self.sub)

    def __getitem__(self, index):
        subpath=self.data_path
        sub=self.sub[index]
        img_from_disk =(str(subpath)+str(sub)+self.name+self.after_sample)
        data_dicts = {"image": str(img_from_disk), "class": int(self.class_base)}
        if self.transform:
            data_dicts_transform = self.transform(data_dicts)
        else:
            data_dicts_transform= data_dicts
        
        return data_dicts_transform
                                    
