#Author: Michail Mamalakis
#Version: 0.1
#Licence:MIT
#email:mm2703@cam.ac.uk

from __future__ import division, print_function

# name of weights store of main segmentation
import nibabel as nib
from scipy import ndimage
import csv
import numpy as np
import os
import torch
import random
from SR_XAI import utils
from SR_XAI.utils import *

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

def excel_label(filenamep='example.csv', given_name='908', cell_col=2, list_name=['absent','prominent','present']):
    with open(filenamep, 'r') as csvfile:
        reader = csv.reader(csvfile)
        cell_value = None
        for row in reader:
            if row[0] == given_name[:-3]: #_T1 we do not need in this case !!
                #print(row[0],given_name[:-3], row[cell_col-1])
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

        #if found_name:
        #print(f"{given_name} was found in the CSV file. The value of cellin column, {cell_col}) is {cell_value}.")
        #else:
            #print(f"{given_name} was not found in the CSV file.")
    csvfile.close()
    return cell_value


def normalize(volume):
    min=np.min(volume)
    max=np.max(volume)
    volume[volume<min]=min
    volume[volume>max]=max
    volume=(volume-min)/(max-min)
    volume=volume.astype("float32")
    return volume

def resize_volume(img,d=112,w=112,h=112):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = d
    desired_width = w
    desired_height = h
    # Get current depth
    current_depth = img.shape[2]
    current_width = img.shape[0]
    current_height = img.shape[1]
		# Compute depth factor
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    #print(img.shape,desired_depth,desired_width,desired_height,width_factor,depth_factor,height_factor)
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

    # Ensure both lists are the same length
    if len(faith) != len(complexity):
        raise ValueError("Both lists must be of the same length.")
    max_faith = max(faith)
    max_complexity = max(complexity)
    normalized_faith = [f / max_faith for f in faith]
    normalized_complexity = [c / max_complexity for c in complexity]

    # Calculate the weights list
    weights = [l1 * f + l2 * (1 / c) for f, c in zip(normalized_faith, normalized_complexity)]
    return weights

def weight_initializer(ex,loader,net):
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
    try:
        data1 = next(exp_iterator0)
    except StopIteration:
        exp_iterator1=iter(ex[0])
        data1 = next(exp_iterator0)
    try:
        data2 = next(exp_iterator1)
    except StopIteration:
        exp_iterator1=iter(ex[1])
        data2 = next(exp_iterator1)
    try:
        data3 = next(exp_iterator2)
    except StopIteration:
        exp_iterator2=iter(ex[2])
        data3 = next(exp_iterator2)
    try:
        data4 = next(exp_iterator3)
    except StopIteration:
        exp_iterator3=iter(ex[3])
        data4 = next(exp_iterator3)
    try:
        data5 = next(exp_iterator4)
    except StopIteration:
        exp_iterator4=iter(ex[4])
        data5 = next(exp_iterator4)
    try:
        data6 = next(exp_iterator5)
    except StopIteration:
        exp_iterator5=iter(ex[5])
        data6 = next(exp_iterator5)
    try:
        data7 = next(exp_iterator6)
    except StopIteration:
        exp_iterator6=iter(ex[6])
        data7 = next(exp_iterator6)
    try:
        data8 = next(exp_iterator7)
    except StopIteration:
        exp_iterator7=iter(ex[7])
        data8 = next(exp_iterator7)
    try:
        data9 = next(exp_iterator8)
    except StopIteration:
        exp_iterator8=iter(ex[8])
        data9 = next(exp_iterator8)

    try:
        data = next(loader_im)
    except StopIteration:
        exp_iterator1=iter(loader)
        data = next(loader_im)

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
            min_com,max_faith=utils.xai_score(imagesv[rdn],y_batchv[rdn],xai_ex,net=net)
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
        #print('mean computing...')
        out=torch.sum(Z, dim=axis) / torch.sum(Wexp.to(device=device, dtype=torch.float)).to(device=device, dtype=torch.float)
    else:
        #list_of_tensors = [torch.tensor(np.array(df)) for df in weight]
        #W=tor
        W=(weight)
        if len(W.shape)==3:
            Wexp=W.view(-1,W.shape[1],W.shape[2],1,1)
        elif len(W.shape)==4:
            Wexp=W.view(-1,W.shape[1],W.shape[2],W.shape[3],1)
        elif len(W.shape)==5:
            Wexp=W.view(-1,W.shape[1],W.shape[2],W.shape[3],W.shape[4])
        else:
            Wexp=W.view(-1,W.shape[1],1,1,1)
         #for i in range(A.shape[0]):
         #    Wexp_batch=torch.stack(Wexp,dim=1)
         #print(Wexp.shape,A.shape)
        Z = torch.mul(A.to(device=device, dtype=torch.float), Wexp.to(device=device, dtype=torch.float))
        #print(Z.shape,A.shape,Wexp.shape,W.shape) ,dim=axis
        out=torch.sum(Z, dim=axis) / torch.sum(Wexp.to(device=device, dtype=torch.float)).to(device=device, dtype=torch.float)
        #print(out.shape,out)
    
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
        items=os.listdir(data_path2+class_base+"/")
        items=sorted(items)
        self.name=name
        o=0
        for item in items:
            index=item.find("i")
            sub_itter=item[:index]
            if str(sub_itter)==str(cs):
                break
            else:
                self.sub.append(sub_itter)
            o=o+1
     
        self.o=o

    def __len__(self):
        return (self.o-1)

    def __getitem__(self, index):
        #image_collection.append(image)
        # Load the image
        subpath=self.data_path
        sub=self.sub[index]
        img_from_disk =(str(subpath)+str(sub)+self.name+self.after_sample)
        data_dicts = {"image": str(img_from_disk), "class": int(self.class_base)}
        if self.transform:
            data_dicts_transform = self.transform(data_dicts)
        else:
            data_dicts_transform= data_dicts
        
        return data_dicts_transform
                                    
