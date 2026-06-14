from __future__ import division, print_function

import os
import torch
import torch.multiprocessing
import torch.nn.functional as nnf
import wandb
from monai.losses import SSIMLoss
from monai.metrics import MultiScaleSSIMMetric

from SR_XAI.models.XAI_SR import *
from SR_XAI.utilities.load_data import *
from SR_XAI.utilities.preprocessing import *
from SR_XAI.utilities.utils import *

device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

def _load_classifier(model_name, classifier_checkpoint=None):
    net2 = networks(model_name, "cpu")
    net = net2.build(c=2)
    checkpoint_path = classifier_checkpoint or os.environ.get("SR_XAI_CLASSIFIER_CHECKPOINT")
    if not checkpoint_path:
        raise ValueError("Set classifier_checkpoint or SR_XAI_CLASSIFIER_CHECKPOINT to the pretrained classifier .pt file.")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    return net

def _init_wandb(project, config):
    api_key = os.environ.get("WANDB_API_KEY")
    mode = os.environ.get("WANDB_MODE", "online" if api_key else "disabled")
    if api_key:
        wandb.login(key=api_key)
    return wandb.init(project=project, config=config, mode=mode)

def train(model='model1',model_name_s="new_upsample_n2norm_sr_model.pt",ep=10,lr=5e-3,roi_size=16,height=112,width=112,depth=112,channels=1,bz=4,DATA_ROOT1='None',DATA_ROOT2='None',PATH='none',PATH1='none',l1=0.3,l2=0.5,l0=0.2,class_base="1",scale=2,min_com=0.001,max_faith=0.96,cs='None',afs='None',names=['DeepLiftshap','KernelShape','Lime','GradShap','Saliency','Intgrag','DeepLift','GuidedBackprop','GuidedGradCam'],nam=['DeepLiftshap','a_batch_KernelShap','a_batch_Lime_','a_batch_gradshap_','a_batch_saliency_','a_batch_intgrad_','a_batch_DeepLift_','a_batch_GuidedBackprop_','GuidedGradCam'],classifier_checkpoint=None):

    batch_n = bz   
    num_epochs = ep
    if model=='attention':
        netsr = XAISR(sr_scale=scale,dim=roi_size).to(device)
    else:
        netsr= XAI_SwinMT(sr_scale=scale,dim=roi_size).to(device)
    loss_function = SSIMLoss(spatial_dims=3,data_range=255,win_size=4)
    issm_acc=MultiScaleSSIMMetric(spatial_dims=3,data_range=255,kernel_size=4)
    optimizer = torch.optim.AdamW(netsr.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(ep*0.85)), gamma=0.1)

    model2='resnet18'
    net=_load_classifier(model2, classifier_checkpoint)


    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    net = net.to(device)
    netsr = netsr.to(device)

    run = _init_wandb("XAI_SR_3D_test", {"learning_rate": lr,"epochs": num_epochs, "batch" : batch_n })
    delta_t=0.0001
    print_every = 1
    mint=0
    minv=10000
    sw=4*bz
    run_roi=1
    checkpoint_path = os.path.join(PATH, model_name_s)
    if os.path.exists(checkpoint_path):
        if torch.cuda.is_available():
            checkpoint = torch.load(checkpoint_path)
            netsr.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            mint = checkpoint['faith']
        else:
            checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
            netsr.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            mint = checkpoint['faith']
            minv=checkpoint['loss_val']
        print('load last saved file: ',PATH, " loss: ",mint, "los_val: ", minv)


    netsr.train()
    print(PATH1)
    explanations,explanations_val,explanations_test= load_saved_xai(roi_size=roi_size,PATH1=PATH1,names=names,bz=batch_n,height=height,width=width,depth=depth,class_base=class_base,cs=cs,afs=afs,nam=nam)

    image_loader, image_loaderv, image_loadert = data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+class_base+'/'),DATA_ROOT2=PATH1,cs=cs,afs=".nii",name="input_image")
    ex_l=explanations.values()
    exv_l=explanations_val.values()
    ex=list(ex_l)
    exv=list(exv_l)  
    assert len(names)==9, "the implementation is for nine xai models:'DeepLiftshap','KernelShape','Lime','GradShap','Saliency','Intgrag','DeepLift','GuidedBackprop','GuidedGradCam' "
    run_loss= AverageMeter()
    run_lossv= AverageMeter()
    run_acc = AverageMeter()
    run_accv = AverageMeter()
    weight=weight_initializer(exv,image_loaderv,net)
    weights=torch.tensor(weight, device=device, dtype=torch.float)
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
        for i,data in enumerate(image_loader):
            base=[]
            output=[]
            data1, exp_iterator0 = next_or_restart(exp_iterator0, ex[0])
            data2, exp_iterator1 = next_or_restart(exp_iterator1, ex[1])
            data3, exp_iterator2 = next_or_restart(exp_iterator2, ex[2])
            data4, exp_iterator3 = next_or_restart(exp_iterator3, ex[3])
            data5, exp_iterator4 = next_or_restart(exp_iterator4, ex[4])
            data6, exp_iterator5 = next_or_restart(exp_iterator5, ex[5])
            data7, exp_iterator6 = next_or_restart(exp_iterator6, ex[6])
            data8, exp_iterator7 = next_or_restart(exp_iterator7, ex[7])
            data9, exp_iterator8 = next_or_restart(exp_iterator8, ex[8])
            y_batch=data["class"].to(device=device,dtype=torch.int)

            images=data["image"].to(device=device,dtype=torch.float)
            xai_images=torch.cat((data1["image"].to(device=device, dtype=torch.float),data2["image"].to(device=device, dtype=torch.float),data3["image"].to(device=device, dtype=torch.float),data4["image"].to(device=device, dtype=torch.float),data5["image"].to(device=device, dtype=torch.float),data6["image"].to(device=device, dtype=torch.float),data7["image"].to(device=device, dtype=torch.float),data8["image"].to(device=device, dtype=torch.float),data9["image"].to(device=device, dtype=torch.float)),1)
           
            xav = weight_mean(xai_images,weights,axis=1)
            base= torch.unsqueeze(xav,1)
            base_t=torch.cat((base,xai_images),dim=1)
            output2, w1= netsr(base_t,images)

            base2=nnf.interpolate(base, size=(int(base.shape[2]*scale),int(base.shape[3]*scale),int(base.shape[4]*scale)), mode='trilinear', align_corners=False)
            output=nnf.interpolate((output2), size=(int(base.shape[2]),int(base.shape[3]),int(base.shape[4])), mode='trilinear', align_corners=False)
            output2t=normalize_tensor(output)
            min_com0,max_faith0=xai_score(images,y_batch,output2t,net=net,device=device)
            loss2=loss_function(output2, base2)
            loss1=loss_function(output, base)
            # Balance reconstruction with the target faithfulness/complexity ranges.
            loss = l0*(0.5*loss1+0.5*loss2)+l1*abs(min_com-min_com0)+l2*abs(max_faith-abs(max_faith0))   
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            issm_acc.reset()

            run_loss.update(loss.item(), n=batch_n)
            issm_acc(output2, base2)
            total_loss_train += loss.item()
            acc=issm_acc.aggregate()
            run_acc.update(acc.item(), n=batch_n)
            metric_f=acc.item()
            total_metric+=metric_f
            labels=base
            accur_metric=(l0*metric_f + l1*safe_ratio(min_com, min_com0) + l2*safe_ratio(max_faith0, max_faith))
            correct_train += accur_metric
            count1=count1+1
            faith_tot+=abs(max_faith0)
            comp_tot+=abs(min_com0)
            
        if count1 == 0:
            raise RuntimeError("Training loader yielded no batches.")
        avg_loss_train = total_loss_train / count1
        accuracy_train = correct_train /count1
        tot_metric=total_metric /count1
        faith=faith_tot/count1
        comp=comp_tot/count1
        
        print("Epoch: ", epoch,"/", num_epochs, " Training Loss: ",avg_loss_train," Train Accuracy: ",accuracy_train," Train Metric: ",tot_metric," Train Faith: ",faith, " Train Comp: ", comp)
        scheduler.step()
        wandb.log({"loss_train": avg_loss_train, "train_faith":faith, " train_comp": comp, "acc_train": accuracy_train, "acc_metric_issm_train": tot_metric})
        if avg_loss_train<=minv and faith>=mint:
            mint=faith 
            minv=avg_loss_train
            print('save model')
            state_dict1 = netsr.state_dict()
            save_dict1 = {'epoch': epoch,'model_state_dict': netsr.state_dict(),'optimizer_state_dict': optimizer.state_dict(),'faith':faith,'loss_val':avg_loss_train,'scheme_state_dict':state_dict1}
            filename1 = os.path.join( PATH,model_name_s)
            torch.save(save_dict1, filename1)
