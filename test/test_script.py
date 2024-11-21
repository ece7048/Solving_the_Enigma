#Author: Michail Mamalakis
#Version: 0.1
#Licence:
#email:mm2703@cam.ac.uk

from __future__ import division, print_function
from SR_XAI import test
from SR_XAI.test import *


epoch=20
lr=1e-3
height=64
width=64
depth=64
channels=1
batch=8
path_fold1='/home/mm2703/rds/hpc-work/data/new_sulci/subjects/'
path_fold2='/home/mm2703/rds/hpc-work/data/new_sulci/subjects/'
model='XAI_SwiftNet'
scale=2
after_samp='_T1align_skel_mask'
PATH='/home/mm2703/rds/hpc-work/code/SR_XAI/'
PATH1='/home/mm2703/rds/hpc-work/data/new_sulci/XAI_QNN/'
l1=0.29
l2=0.6
l0=0.01
max_faith=0.9
min_comp=9.0
cs="37_16"
afs='explanation.nii'
names=['DeepLiftshap','KernelShape','Lime','GradShap','Saliency','Intgrag','DeepLift','GuidedBackprop','GuidedGradCam']
class_base="1"
model_name="training_sr_swift_v1.pt"
nam=['DeepLiftshap','a_batch_KernelShap','a_batch_Lime_','a_batch_gradshap_','a_batch_saliency_','a_batch_intgrad_','a_batch_DeepLift_','a_batch_GuidedBackprop_','GuidedGradCam']
save=True
r_s=64

test(model, epoch, lr, r_s,height, width, depth, channels, batch, path_fold1, path_fold2, PATH, PATH1,l1,l2,l0, class_base, scale, min_comp,max_faith,cs,afs,names,nam,save,model_name)
