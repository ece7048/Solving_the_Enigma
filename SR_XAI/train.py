#Author: Michail Mamalakis
#Version: 0.1
#Licence:MIT

from __future__ import division, print_function
from SR_XAI import load_data,preprocessing ,XAI_SR, utils,monai_utils
from SR_XAI.load_data import *
from SR_XAI.XAI_SR import *
from SR_XAI.utils import *
from SR_XAI.monai_utils import *
from SR_XAI.preprocessing import *

import torch.multiprocessing
from functools import partial
from monai.data import decollate_batch

# train the model
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import os
import wandb
import torchvision
import torchvision.transforms as T
import torch.nn.functional as nnf
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
import random
from monai.losses import DiceLoss,SSIMLoss
from monai.metrics import DiceMetric,ROCAUCMetric,HausdorffDistanceMetric,ConfusionMatrixMetric,MSEMetric,MultiScaleSSIMMetric

#warnings.filterwarnings("ignore")
#device = torch.device( "cpu")
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')


def train(model='model1',model_name_s="new_upsample_n2norm_sr_model.pt",ep=10,lr=5e-3,roi_size=16,height=112,width=112,depth=112,channels=1,bz=4,DATA_ROOT1='None',DATA_ROOT2='None',PATH='none',PATH1='none',l1=0.4,l2=0.4,l0=0.2,class_base="1",scale=2,min_com=0.001,max_faith=0.96,cs='None',afs='None',names=['DeepLiftshap','KernelShape','Lime','GradShap','Saliency','Intgrag','DeepLift','GuidedBackprop','GuidedGradCam'],nam=['DeepLiftshap','a_batch_KernelShap','a_batch_Lime_','a_batch_gradshap_','a_batch_saliency_','a_batch_intgrad_','a_batch_DeepLift_','a_batch_GuidedBackprop_','GuidedGradCam']):

    batch_n = bz   
    num_epochs = ep
    if model=='attention':
        netsr = XAISR(sr_scale=scale,dim=roi_size).to(device)
    else:
        netsr= XAI_SwinMT(sr_scale=scale,dim=roi_size).to(device)
    loss_function = SSIMLoss(spatial_dims=3,data_range=255,win_size=4)
    issm_acc=MultiScaleSSIMMetric(spatial_dims=3,data_range=255,kernel_size=4)
    optimizer = torch.optim.AdamW(netsr.parameters(), lr=lr, weight_decay=1e-4)
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(ep*0.85), gamma=0.1)

    model2='resnet18'
    net2=networks(model2,"cpu")
    net=net2.build(c=2)
    PATH3='/home/mm2703/rds/hpc-work/code/QCNN/'+model2+'_L_no_norm_crop_resize'
    checkpoint = torch.load(PATH3+".pt",map_location=torch.device(device))
    net.load_state_dict(checkpoint['model_state_dict'])


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #net = nn.DataParallel(net)
        #netsr = nn.DataParallel(netsr)
    net = net.to(device)
    netsr = netsr.to(device)
    wandb.login(key="5d50fdf4c31cf52cf6e5789c6a26b93311d1648c")

    run = wandb.init(# Set the project where this run will be logged
    project="XAI_SR_3D_test", #"prova-project-ovarian"
    # Track hyperparameters and run metadata
    config={"learning_rate": 5e-3,"epochs": num_epochs, "batch" : batch_n })
    delta_t=0.0001
    print_every = 1
    mint=0
    minv=10000
    sw=4*bz
    run_roi=1 #int(2*8)
    #model_inferer = partial(sliding_window_inference,roi_size=[roi_size,int(roi_size/1),int(roi_size/1)],sw_batch_size=sw,predictor=netsr,overlap=0.5)
    if os.path.exists(PATH+model_name_s):
        if torch.cuda.is_available():
            checkpoint = torch.load(PATH+model_name_s)
            netsr.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            mint = checkpoint['faith']
         #    minv=checkpoint['loss_val']
        else:
            checkpoint = torch.load(PATH+model_name_s,map_location=torch.device('cpu'))
            netsr.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            mint = checkpoint['faith']
            minv=checkpoint['loss_val']
        print('load last saved file: ',PATH, " loss: ",mint, "los_val: ", minv)


    netsr.train()  # Set the model in training mode
    print(PATH1)
    explanations,explanations_val,explanations_test= load_saved_xai(roi_size=roi_size,PATH1=PATH1,names=names,bz=batch_n,height=height,width=width,depth=depth,class_base=class_base,cs=cs,afs=afs,nam=nam)

    image_loader, image_loaderv, image_loadert = data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+class_base+'/'),DATA_ROOT2=PATH1,cs=cs,afs=".nii",name="input_image")
    ex_l=explanations.values()
    exv_l=explanations_val.values()
    ex=list(ex_l)
    exv=list(exv_l)  
    #ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    assert len(names)==9, "the implementation is for nine xai models:'DeepLiftshap','KernelShape','Lime','GradShap','Saliency','Intgrag','DeepLift','GuidedBackprop','GuidedGradCam' "
    run_loss= AverageMeter()
    run_lossv= AverageMeter()
    run_acc = AverageMeter()
    run_accv = AverageMeter()
    weight=weight_initializer(exv,image_loaderv,net)
    weights=torch.tensor(weight)
    print('The weights of this mean value is: ',weights)
    for epoch in range(num_epochs):       
        netsr.train()
        exp_iterator0=iter(ex[0])
        exp_iterator1=iter(ex[1])
        exp_iterator2=iter(ex[2])
        exp_iterator3=iter(ex[3])
        exp_iterator4=iter(ex[4])
        exp_iterator5=iter(ex[5])
        exp_iterator6=iter(ex[6])
        exp_iterator7=iter(ex[7])
        exp_iterator8=iter(ex[8])
        count1=0
        total_loss_train = 0
        correct_train = 0
        total_train = 0
        total_metric =0
        faith_tot=0
        comp_tot=0
        netsr.train()
        base=[]
        output=[]
        loss=0
        for i,data in enumerate(image_loader):
            base=[]
            output=[]
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
            y_batch=data["class"].to(device=device,dtype=torch.int)

            images=data["image"].to(device=device,dtype=torch.float)
            xai_images=torch.cat((data1["image"].to(device=device, dtype=torch.float),data2["image"].to(device=device, dtype=torch.float),data3["image"].to(device=device, dtype=torch.float),data4["image"].to(device=device, dtype=torch.float),data5["image"].to(device=device, dtype=torch.float),data6["image"].to(device=device, dtype=torch.float),data7["image"].to(device=device, dtype=torch.float),data8["image"].to(device=device, dtype=torch.float),data9["image"].to(device=device, dtype=torch.float)),1)
           
            xav = weight_mean(xai_images,weights,axis=1)
            base= torch.unsqueeze(xav,1)
            base_t=torch.cat((base,xai_images),dim=1)
            output2, w1= netsr(base_t,images)

            base2=nnf.interpolate(base, size=(int(base.shape[2]*scale),int(base.shape[3]*scale),int(base.shape[4]*scale)), mode='trilinear', align_corners=False)
            output=nnf.interpolate((output2), size=(int(base.shape[2]),int(base.shape[3]),int(base.shape[4])), mode='trilinear', align_corners=False)
            output2t=((output-torch.min(output))/(torch.max(output)-(torch.min(output))))*255
            min_com0,max_faith0=xai_score(images,y_batch,output2t,net=net,device=device)
            loss2=loss_function(output2, base2)
            loss1=loss_function(output, base)
            loss += l0*(0.2*loss1+0.8*loss2)+l1*abs(min_com-min_com0)+l2*abs(max_faith-abs(max_faith0))
            issm_acc.reset()

            run_loss.update(loss.item(), n=batch_n)
            loss_p=run_loss.avg
            issm_acc(output2, base2)
            total_loss_train = loss_p
            acc=issm_acc.aggregate()
            run_acc.update(acc.item(), n=batch_n)
            metric_f=run_acc.avg
            total_metric+=metric_f
            labels=base
            # Backward pass
            accur_metric=(l0*metric_f + l1*abs(min_com/min_com0) + l2*abs(max_faith0/max_faith))
            # Calculate training accuracyfV
            correct_train += accur_metric
            #print(f"ImagesBatch ->  Training Loss: {total_loss_train:.4f}, Train Accuracy: {correct_train:.4f}")
            count1=count1+1
            faith_tot+=abs(max_faith0)
            comp_tot+=abs(min_com0)
            #print("the weights of this batch are: ",w1.shape,w1)
            
        loss_f=(loss/(count1*batch_n))
        loss_f.backward()#retain_graph=True)
        optimizer.step()
        avg_loss_train = total_loss_train / count1#/ len(image_loader)
        accuracy_train = correct_train /count1 #/ len(image_loader)
        tot_metric=total_metric /count1 #/ len(image_loader)
        faith=faith_tot/count1
        comp=comp_tot/count1
        
        #netsr.eval()  # Set the model in evaluation mode
        print("Epoch: ", epoch,"/", num_epochs, " Training Loss: ",avg_loss_train," Train Accuracy: ",accuracy_train," Train Metric: ",tot_metric," Train Faith: ",faith, " Train Comp: ", comp)
        scheduler.step()
        #print("the last batch weights of this epoch are: ",w1.shape,w1)
        wandb.log({"loss_train": avg_loss_train, "train_faith":faith, " train_comp": comp, "acc_train": accuracy_train, "acc_metric_issm_train": tot_metric})
        if avg_loss_train<=minv and faith>=mint:
            mint=faith 
            minv=avg_loss_train
            print('save model')
            state_dict1 = netsr.state_dict()
            save_dict1 = {'epoch': epoch,'model_state_dict': netsr.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'faith':faith,'loss_val':avg_loss_train,'scheme_state_dict':state_dict1}
            filename1 = os.path.join( PATH,model_name_s)
            torch.save(save_dict1, filename1)

