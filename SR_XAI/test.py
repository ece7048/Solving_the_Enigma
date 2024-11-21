#Author: Michail Mamalakis
#Version: 0.1
#Licence:MIT

from __future__ import division, print_function
from SR_XAI import load_data, XAI_SR, utils, monai_utils,preprocessing
from SR_XAI.load_data import *
from SR_XAI.XAI_SR import *
from SR_XAI.utils import *
from SR_XAI.monai_utils import *
from SR_XAI.preprocessing import *

import torch.multiprocessing

# train the model
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
import os
import wandb
import torchvision
import torchvision.transforms as T
import torch.nn.functional as nnf
from torchmetrics.image import StructuralSimilarityIndexMeasure
import random
from monai.losses import DiceLoss,SSIMLoss
from monai.metrics import DiceMetric,ROCAUCMetric,HausdorffDistanceMetric,ConfusionMatrixMetric,MSEMetric,MultiScaleSSIMMetric
from monai.data import decollate_batch
from functools import partial
import torch.nn as nn

#warnings.filterwarnings("ignore")
#device = torch.device( "cpu")
device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')


def test(model='model1',ep=10,lr=5e-3,roi_size=16,height=112,width=112,depth=112,channels=1,bz=4,DATA_ROOT1='None',DATA_ROOT2='None',PATH='none',PATH1='none',l1=0.4,l2=0.4,l0=0.2,class_base="1",scale=2,min_com=0.001,max_faith=0.96,cs='None',afs='None',names=['DeepLiftshap','KernelShape','Lime','GradShap','Saliency','Intgrag','DeepLift','GuidedBackprop','GuidedGradCam'],nam=['DeepLiftshap','a_batch_KernelShap','a_batch_Lime_','a_batch_gradshap_','a_batch_saliency_','a_batch_intgrad_','a_batch_DeepLift_','a_batch_GuidedBackprop_','GuidedGradCam'], save_nii=True, model_name="sr_model_whole.pt"):

    batch_n = bz   
    num_epochs = ep
    if model=='attention':
        netsr = XAISR(sr_scale=scale,dim=roi_size).to(device)
    else:
        netsr= XAI_SwinMT(sr_scale=scale,dim=roi_size).to(device)
    #loss_function = SSIMLoss(spatial_dims=3,data_range=255)
    #issm_acc=MultiScaleSSIMMetric(spatial_dims=3,data_range=255,kernel_size=4)
    optimizer = torch.optim.AdamW(netsr.parameters(), lr=lr, weight_decay=1e-4)
    #scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

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
    config={ "batch" : batch_n })

    if os.path.exists(PATH+model_name):
        if torch.cuda.is_available():
            checkpoint = torch.load(PATH+model_name)
            netsr.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            mint = checkpoint['faith']
            minv=checkpoint['loss_val']
        else:
            checkpoint = torch.load(PATH+model_name,map_location=torch.device('cpu'))
            netsr.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            mint = checkpoint['faith']
            minv=checkpoint['loss_val']
        print('load last saved file: ',PATH, " loss: ",mint, "los_val: ", minv)


    netsr.train()  # Set the model in training mode
    print(PATH1)
    explanations,explanations_val,explanations_test= load_saved_xai(roi_size=roi_size,PATH1=PATH1,names=names,bz=batch_n,height=height,width=width,depth=depth,class_base=class_base,cs=cs,afs=afs,nam=nam)

    image_loader, image_loaderv, image_loadert = data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+class_base+'/'),DATA_ROOT2=PATH1,cs=cs,afs=".nii",name="input_image")
    exv_v=explanations_val.values()
    exv=list(exv_v)  

    weight=weight_initializer(exv,image_loaderv,net)
    weights=torch.tensor(weight)
    #ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
    assert len(names)==9, "the implementation is for nine xai models:'DeepLiftshap','KernelShape','Lime','GradShap','Saliency','Intgrag','DeepLift','GuidedBackprop','GuidedGradCam' "

    if save_nii==True:
        prefix_store_path=PATH+'/test_samples/'
        store_path=PATH+'/test_samples/valid_results/'
        store_path2=PATH+'/test_samples/test_results'
        if os.path.isdir(prefix_store_path):
            print('folders exist')
        else:
            os.mkdir(prefix_store_path)
        if os.path.isdir(store_path):
            print('folders exist')
        else:
            os.mkdir(store_path)
        if os.path.isdir(store_path2):
            print('folders exist')
        else:
            os.mkdir(store_path2)
    print('The training results are: ')
    ex_=explanations.values()
    ex=list(ex_)
    #compute(image_loader,weights,ex,netsr,scale,net,l0,l1,l2,batch_n,roi_size,min_com,max_faith,prefix_store_path)

    print('The validation results are: ')
    #compute(image_loaderv,weights,exv,netsr,scale,net,l0,l1,l2,batch_n,roi_size,min_com,max_faith,store_path)

    exv_t=explanations_test.values()
    ext=list(exv_t)    
    print('The test results are: ')   
    compute(image_loadert,weights,ext,netsr,scale,net,l0,l1,l2,batch_n,roi_size,min_com,max_faith,store_path2,False)    
    print('end')




def compute(image_loadert,weights,exv,netsr,scale,net,l0,l1,l2,batch_n,roi_size=16,min_com=0.001,max_faith=0.96,store="",save=True):
    print_every = 1
    mint=10000
    minv=10000
    loss_function = SSIMLoss(spatial_dims=3,data_range=255)
    issm_acc=MultiScaleSSIMMetric(spatial_dims=3,data_range=255,kernel_size=4)
    run_loss= AverageMeter()
    run_lossv= AverageMeter()
    run_acc = AverageMeter()
    run_accv = AverageMeter()
    f1_t=0
    c1_t=0
    f2_t=0
    c2_t=0
    f3_t=0
    c3_t=0
    f4_t=0
    c4_t=0
    f5_t=0
    c5_t=0
    f6_t=0
    c6_t=0
    f7_t=0
    c7_t=0
    f8_t=0
    c8_t=0
    f9_t=0
    c9_t=0
    fb_t=0
    cb_t=0
    test=True
    sw=4*batch_n
   # model_inferer = partial(sliding_window_inference,roi_size=[roi_size,int(roi_size/2),int(roi_size/8)],sw_batch_size=sw,predictor=netsr,overlap=0.0)
    if test==True:
        netsr.eval()  
        total_loss_trainv = 0
        correct_trainv = 0
        total_trainv = 0
        total_metricv=0
        comp_totv=0
        faith_totv=0
        count2=0
        with torch.no_grad():
            exp_iterator0v=iter(exv[0])
            exp_iterator1v=iter(exv[1])
            exp_iterator2v=iter(exv[2])
            exp_iterator3v=iter(exv[3])
            exp_iterator4v=iter(exv[4])
            exp_iterator5v=iter(exv[5])
            exp_iterator6v=iter(exv[6])
            exp_iterator7v=iter(exv[7])
            exp_iterator8v=iter(exv[8])
            
            for i,data in enumerate(image_loadert):
                try:
                    data1 = next(exp_iterator0v)
                except StopIteration:
                    exp_iterator0v=iter(exv[0])
                    data1 = next(exp_iterator0v)
                try:
                    data2 = next(exp_iterator1v)
                except StopIteration:
                    exp_iterator1v=iter(exv[1])
                    data2 = next(exp_iterator1v)
                try:
                    data3 = next(exp_iterator2v)
                except StopIteration:
                    exp_iterator2v=iter(exv[2])
                    data3 = next(exp_iterator2v)
                try:
                    data4 = next(exp_iterator3v)
                except StopIteration:
                    exp_iterator3v=iter(exv[3])
                    data4 = next(exp_iterator3v)
                try:
                    data5 = next(exp_iterator4v)
                except StopIteration:
                    exp_iterator4v=iter(exv[4])
                    data5 = next(exp_iterator4v)
                try:
                    data6 = next(exp_iterator5v)
                except StopIteration:
                    exp_iterator5v=iter(exv[5])
                    data6 = next(exp_iterator5v)
                try:
                    data7 = next(exp_iterator6v)
                except StopIteration:
                    exp_iterator6v=iter(exv[6])
                    data7 = next(exp_iterator6v)
                try:
                    data8 = next(exp_iterator7v)
                except StopIteration:
                    exp_iterator7v=iter(exv[7])
                    data8 = next(exp_iterator7v)
                try:
                    data9 = next(exp_iterator8v)
                except StopIteration:
                    exp_iterator8v=iter(exv[8])
                    data9 = next(exp_iterator8v)

                imagesv=data["image"].to(device=device,dtype=torch.float)
                y_batchv=data["class"].to(device=device,dtype=torch.int)

                xai_images_v=torch.cat((data1["image"].to(device=device, dtype=torch.float),data2["image"].to(device=device, dtype=torch.float),data3["image"].to(device=device, dtype=torch.float),data4["image"].to(device=device, dtype=torch.float),data5["image"].to(device=device, dtype=torch.float),data6["image"].to(device=device, dtype=torch.float),data7["image"].to(device=device, dtype=torch.float),data8["image"].to(device=device, dtype=torch.float),data9["image"].to(device=device, dtype=torch.float)),1)




                # Forward pass
                #outputvs= model_inferer(inputs=xai_images_v,inputs2=imagesv)
                xav = weight_mean(xai_images_v,weights,axis=1)
                basev= torch.unsqueeze(xav,1)
                basev_t=torch.cat((basev,xai_images_v),dim=1)
                #outputvs= model_inferer(inputs=basev_t,inputs2=imagesv)
                #outputlvs = decollate_batch(outputvs)
                #outputvo = [(val_pred_tensorgtv) for val_pred_tensorgtv in outputlvs]
                #outputv=torch.stack(outputvo,dim=0)
                outputvs, _=netsr(basev_t,imagesv)
                outputv=outputvs
                basev=nnf.interpolate(basev, size=(int(imagesv.shape[3]*scale),int(imagesv.shape[3]*scale),int(imagesv.shape[3]*scale)), mode='trilinear', align_corners=False)
                output2v=nnf.interpolate(outputv, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                output2v=((output2v-torch.min(output2v))/(torch.max(output2v)-(torch.min(output2v))))*255
                #output21v=((output2v-torch.min(outputv))/(torch.max(outputv)-(torch.min(outputv))))
                #GTv=((basev-torch.min(basev))/(torch.max(basev)-(torch.min(basev))))
                rdn=random.randint(0,imagesv.shape[0]-1)
                data1=data1["image"].to(device=device, dtype=torch.float)
                data2=data2["image"].to(device=device, dtype=torch.float)
                data3=data3["image"].to(device=device, dtype=torch.float)
                data4=data4["image"].to(device=device, dtype=torch.float)
                data5=data5["image"].to(device=device, dtype=torch.float)
                data6=data6["image"].to(device=device, dtype=torch.float)
                data7=data7["image"].to(device=device, dtype=torch.float)
                data8=data8["image"].to(device=device, dtype=torch.float)
                data9=data9["image"].to(device=device, dtype=torch.float)
                

                data1=nnf.interpolate(data1, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                data1=((data1-torch.min(data1))/(torch.max(data1)-(torch.min(data1))))*255
                data2=nnf.interpolate(data2, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                data2=((data2-torch.min(data2))/(torch.max(data2)-(torch.min(data2))))*255
                data3=nnf.interpolate(data3, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                data3=((data3-torch.min(data3))/(torch.max(data3)-(torch.min(data3))))*255
                data4=nnf.interpolate(data4, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                data4=((data4-torch.min(data4))/(torch.max(data4)-(torch.min(data4))))*255
                data5=nnf.interpolate(data5, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                data5=((data5-torch.min(data5))/(torch.max(data5)-(torch.min(data5))))*255
                data6=nnf.interpolate(data6, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                data6=((data6-torch.min(data6))/(torch.max(data6)-(torch.min(data6))))*255
                data7=nnf.interpolate(data7, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                data7=((data7-torch.min(data7))/(torch.max(data7)-(torch.min(data7))))*255
                data8=nnf.interpolate(data8, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                data8=((data8-torch.min(data8))/(torch.max(data8)-(torch.min(data8))))*255
                data9=nnf.interpolate(data9, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                data9=((data9-torch.min(data9))/(torch.max(data9)-(torch.min(data9))))*255
                basev1=nnf.interpolate(basev, size=(int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3])), mode='trilinear', align_corners=False)
                basev1=((basev1-torch.min(basev1))/(torch.max(basev1)-(torch.min(basev1))))*255
     
                min_com0v,max_faith0v=xai_score(imagesv[rdn],y_batchv[rdn],output2v[rdn],net=net,device=device)
                c1,f1=xai_score(imagesv[rdn],y_batchv[rdn],data1[rdn],net=net)
                c2,f2=xai_score(imagesv[rdn],y_batchv[rdn],data2[rdn],net=net)
                c3,f3=xai_score(imagesv[rdn],y_batchv[rdn],data3[rdn],net=net)
                c4,f4=xai_score(imagesv[rdn],y_batchv[rdn],data4[rdn],net=net)
                c5,f5=xai_score(imagesv[rdn],y_batchv[rdn],data5[rdn],net=net)
                c6,f6=0,0#xai_score(imagesv[rdn],y_batchv[rdn],data6[rdn],net=net)
                c7,f7=xai_score(imagesv[rdn],y_batchv[rdn],data7[rdn],net=net)
                c8,f8=xai_score(imagesv[rdn],y_batchv[rdn],data8[rdn],net=net)
                c9,f9=xai_score(imagesv[rdn],y_batchv[rdn],data9[rdn],net=net)
                cb,fb=xai_score(imagesv[rdn],y_batchv[rdn],basev1[rdn],net=net)
                #outputv = torch.stack(list(outputv), dim=0)
                # Compute loss
                loss1v=loss_function(outputv, basev)
                lossv = l0*loss1v+l1*abs(min_com-min_com0v)+l2*abs(max_faith-abs(max_faith0v))
                run_lossv.update(lossv.item(), n=batch_n)
                loss_pv=run_lossv.avg
                total_loss_trainv += loss_pv
                labelsv=basev
                # Calculate training accuracyfV                            
                issm_acc.reset()
                issm_acc(outputv, basev)
                accv=issm_acc.aggregate()
                run_accv.update(accv.item(), n=batch_n)
                metric_fv=run_accv.avg
                acc_metv=(l0*metric_fv+l1*abs(min_com/min_com0v)+l2*abs(max_faith0v/max_faith))

                total_metricv += metric_fv
                correct_trainv += acc_metv
                faith_totv += abs(max_faith0v)
                comp_totv += abs(min_com0v)

                c1_t+=abs(c1)
                f1_t+=abs(f1)
                c2_t+=abs(c2)
                f2_t+=abs(f2)
                c3_t+=abs(c3)
                f3_t+=abs(f3)
                c4_t+=abs(c4)
                f4_t+=abs(f4)
                c5_t+=abs(c5)
                f5_t+=abs(f5)
                c6_t+=abs(c6)
                f6_t+=abs(f6)
                c7_t+=abs(c7)
                f7_t+=abs(f7)
                c8_t+=abs(c8)
                f8_t+=abs(f8)
                c9_t+=abs(c9)
                f9_t+=abs(f9)
                cb_t+=abs(cb)
                fb_t+=abs(fb)
                                               
                if (count2%3==0) and (save==True):
                    outputv1=((outputv-torch.min(outputv))/(torch.max(outputv)-(torch.min(outputv))))*255
                    sp=(1,int(imagesv.shape[3]*scale),int(imagesv.shape[3]*scale),int(imagesv.shape[3]*scale))
                    sp2=(1,int(imagesv.shape[3]),int(imagesv.shape[3]),int(imagesv.shape[3]))
                    save((store+str(count2)+'_batch_first_SR_prediction.nii'),outputv1[0].cpu().numpy(),sp)
                    save((store+str(count2)+'_batch_first_data1.nii'),data1[0].cpu().numpy(),sp2)
                    save((store+str(count2)+'_batch_first_data2.nii'),data2[0].cpu().numpy(),sp2)
                    save((store+str(count2)+'_batch_first_data3.nii'),data3[0].cpu().numpy(),sp2)
                    save((store+str(count2)+'_batch_first_data4.nii'),data4[0].cpu().numpy(),sp2)
                    save((store+str(count2)+'_batch_first_data5.nii'),data5[0].cpu().numpy(),sp2)
                    save((store+str(count2)+'_batch_first_data6.nii'),data6[0].cpu().numpy(),sp2)
                    save((store+str(count2)+'_batch_first_data7.nii'),data7[0].cpu().numpy(),sp2)
                    save((store+str(count2)+'_batch_first_data8.nii'),data8[0].cpu().numpy(),sp2)
                    save((store+str(count2)+'_batch_first_data9.nii'),data9[0].cpu().numpy(),sp2)
                    save((store+str(count2)+'_batch_first_prediction_mean.nii'),basev1[0].cpu().numpy(),sp2)
                    save((store+str(count2)+'_batch_first_prediction.nii'),output2v[0].cpu().numpy(),sp2)
              
                count2=count2+1
                
        faithv=faith_totv/count2
        compv=comp_totv/count2
        c1_tv=c1_t/count2
        f1_tv=f1_t/count2
        c2_tv=c2_t/count2
        f2_tv=f2_t/count2
        c3_tv=c3_t/count2
        f3_tv=f3_t/count2
        c4_tv=c4_t/count2
        f4_tv=f4_t/count2
        c5_tv=c5_t/count2
        f5_tv=f5_t/count2
        c6_tv=c6_t/count2
        f6_tv=f6_t/count2
        c7_tv=c7_t/count2
        f7_tv=f7_t/count2
        c8_tv=c8_t/count2
        f8_tv=f8_t/count2
        c9_tv=c9_t/count2
        f9_tv=f9_t/count2
        cb_tv=cb_t/count2
        fb_tv=fb_t/count2
        avg_loss_trainv = total_loss_trainv /count2
        accuracy_trainv = correct_trainv / count2
        tot_metricv=total_metricv/ count2

        
        print( " Validation Loss: ",avg_loss_trainv," Validation Accuracy: ",accuracy_trainv," Validation Metric: ",tot_metricv," Test Faith: ",faithv, " Test Comp: ", compv, " f1: ",f1_tv, " c1: ",c1_tv, " f2: ",f2_tv, " c2: ",c2_tv, " f3: ",f3_tv, " c3: ",c3_tv, " f4: ",f4_tv, " c4: ",c4_tv, " f5: ",f5_tv, " c5: ",c5_tv, " f6: ",f6_tv, " c6: ",c6_tv, " f7: ",f7_tv, " c7: ",c7_tv, " f8: ",f8_tv, " c8: ",c8_tv, " f9: ",f9_tv, " c9: ",c9_tv, " fb: ",fb_tv, " cb: ",cb_tv)

        wandb.log({"loss_test": avg_loss_trainv, "test_faith": faithv, "test_comp": compv, "acc_test": accuracy_trainv, "acc_metric_issm_test": tot_metricv, "aver_faith": fb_t, "aver_comp": cb_t})

     
