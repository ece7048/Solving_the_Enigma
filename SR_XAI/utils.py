#Author: Michail Mamalakis
#Version: 0.1
#Licence:
from __future__ import division, print_function
from SR_XAI import load_data
from SR_XAI.load_data import *
import csv
import quantus
import nibabel as nib
import numpy as np
import jax
import jax.numpy as jnp

from monai.transforms import (
    AsDiscrete,
    Activations,
)
from monai.transforms import (
    AsDiscrete,
    Compose)

def save(store,volume,sp=(1,112,112,112)):
    volume=np.reshape(volume,(sp[1],sp[2],sp[3],sp[0]))
    volume=volume
    imgnthree1=nib.Nifti1Image(volume,affine=np.eye(4))
    imgnthree1.header.set_data_dtype(np.uint16)
    imgnthree1.header.set_sform(affine=np.eye(4),code='talairach')
    nib.save(imgnthree1,store)

def load_saved_xai(roi_size=16,PATH1='',names=['DeepLiftshap','KernelShape','Lime','GradShap','Saliency','Intgrag','DeepLift','GuidedBackprop','GuidedGradCam'],bz=2,height=112,width=112,depth=112,class_base="1",cs='None',afs='None',nam=['DeepLiftshap','a_batch_KernelShap','a_batch_Lime_','a_batch_gradshap_','a_batch_saliency_','a_batch_intgrad_','a_batch_DeepLift_','a_batch_GuidedBackprop_','GuidedGradCam']):

    sal_loader, sal_loaderv, sal_loadert = data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+names[4]+'/'),DATA_ROOT2=PATH1,cs=cs,afs=afs,name=nam[4])
    gradshap, gradshapv ,gradshapt = data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+names[3]+'/'),DATA_ROOT2=PATH1,cs=cs,afs=afs,name=nam[3])
    intgrad,intgradv, intgradt = data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+names[5]+'/'),DATA_ROOT2=PATH1,cs=cs,afs=afs,name=nam[5])
    DeepLift, DeepLiftv, DeepLiftt = data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+names[6]+'/'),DATA_ROOT2=PATH1,cs=cs,afs=afs,name=nam[6])
    GuidedBackprop, GuidedBackpropv, GuidedBackpropt = data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+names[7]+'/'),DATA_ROOT2=PATH1,cs=cs,afs=afs,name=nam[7])
    Lime, Limev, Limet= data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+names[2]+'/'),DATA_ROOT2=PATH1,cs=cs,afs=afs,name=nam[2])
    KernelShap, KernelShapv, KernelShapt = data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+names[1]+'/'),DATA_ROOT2=PATH1,cs=cs,afs=afs,name=nam[1])
    DeepLiftShap, DeepLiftShapv, DeepLiftShapt,= data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+names[0]+'/'),DATA_ROOT2=PATH1,cs=cs,afs=afs,name=nam[0])
    GuidedGradCam, GuidedGradCamv, GuidedGradCamt,= data_build(roi_size=roi_size,bz=bz,sx=height,sy=width,sz=depth,class_base=class_base,DATA_ROOT1=(PATH1+names[8]+'/'),DATA_ROOT2=PATH1,cs=cs,afs=afs,name=nam[8])

    explanations = {"Saliency": sal_loader, "GradientShap": gradshap,"IntegratedGradients": intgrad,"DeepLift": DeepLift,
    "GuidedBackprop": GuidedBackprop,"LIME": Lime,"KernelShap": KernelShap,"DeepLiftShap": DeepLiftShap,"GuidedGradCam": GuidedGradCam}
    explanations_val = {"Saliency": sal_loaderv, "GradientShap": gradshapv,"IntegratedGradients": intgradv,"DeepLift": DeepLiftv,
    "GuidedBackprop": GuidedBackpropv,"LIME": Limev,"KernelShap": KernelShapv,"DeepLiftShap": DeepLiftShapv, "GuidedGradCam": GuidedGradCamv}
    explanations_test = {"Saliency": sal_loadert, "GradientShap": gradshapt,"IntegratedGradients": intgradt,"DeepLift": DeepLiftt,
    "GuidedBackprop": GuidedBackpropt,"LIME": Limet,"KernelShap": KernelShapt,"DeepLiftShap": DeepLiftShapt, "GuidedGradCam": GuidedGradCamt}

    return explanations,explanations_val,explanations_test

def xai_score(x,y,a,net,device = torch.device( "cpu")):
#    print(x.shape,y.shape)
#    print(y)
    score_c,score_f,score_f1=[],[],[]
    if len(x.shape)==4:
        x=torch.unsqueeze(x,0)
        a=torch.unsqueeze(a,0)
        y=torch.unsqueeze(y,0)
    x_b=x#.permute((4,0,1,2,3))
    a_b=a
    y_b=y
    net.eval()
    net.to(device) 
    for i in range(x_b.shape[1]):
        x_=x_b[:,:,:,:,:].reshape(x_b.shape[0],1,x_b.shape[2],x_b.shape[3],x_b.shape[4])
        x_1=x_b[:,:,:,:,:] #:,i,:,:,:
        x_1=x_1.reshape(x_b.shape[0],1,x_b.shape[2],x_b.shape[3],x_b.shape[4])
        a=a_b[:,:,:,:,:].reshape(a_b.shape[0],a_b.shape[2],a_b.shape[3],a_b.shape[4],1)
        a1=a_b[:,:,:,:,:] #:,i,:,:,:
        a1=a1.reshape(a_b.shape[0],1,a_b.shape[2],a_b.shape[3],a_b.shape[4]) #2, 0
        a_=a.cpu()
        a_1=a1.detach().cpu().numpy() 
        y_=y_b[:].reshape(x_b.shape[0],1)
        y_o=y_.detach().cpu().numpy()
        x_11=x_1.detach().cpu().numpy()
    #    print(x_.shape,a_.shape,y_.shape)
    #    print(x_11.shape,a_1.shape,y_o.shape)

    #Complexity
    # Return complexity scores in an one-liner - by calling the metric instance.
        rc=quantus.Complexity(disable_warnings=True)(model=net, x_batch=x_,y_batch=y_,a_batch=a_.detach().numpy(),device=device,channel_first=True)
        rca=np.average(rc)

    rd=quantus.FaithfulnessCorrelation(
    nr_runs=40,  
    subset_size=64,  
    perturb_baseline="black",
    perturb_func=quantus.perturb_func.baseline_replacement_by_indices,
    similarity_func=quantus.similarity_func.correlation_pearson,  
    abs=False,  
    return_aggregate=False, disable_warnings = True)(model=net,x_batch=x_11, y_batch=y_o,a_batch=a_1,channel_first=True,device=device)
    rda=np.average(rd)
    # print('faithfulness: ',rda)
    complexity=rca
    faithfulness=rda
    return complexity, faithfulness

#averaging
def test_comp(x_batch,y_batch,explanations, bz=2, device = torch.device( "cpu")):
  com_t,faith_t=[],[]
  x=x_batch
  y=y_batch
  total_com,total_faith,total_suf,exp_np=[],[],[],[]
  length=len(explanations)

  for i in range(1,length):
      ex=explanations.values()
      exp=list(ex)[i-1]
      for o, data in enumerate(exp):
        exp_one=data[0]
        for n in range(bz-1):
            val=(exp_one[n,:,:,:,:].reshape(1,1,112,112,112))
            if o==0:
                val_tot=val
            else:
                val_tot=np.append(val,val_tot,axis=0)
      exp_=torch.tensor(val_tot)
      xm=x[:(bz)].reshape(bz,1,112,112,112)
      ym=y[:(bz)].reshape(bz,1)
      com,faith,suf=xai_score(xm,ym,exp_, device = device)
      average=np.average(val_tot,axis=0)
      com_average=np.average(com)
      faith_average=np.average(faith)
      suf_average=np.average(suf)
      total_faith.append(faith_average)
      total_com.append(com_average)
      total_suf.append(suf_average)

  total_faith=np.array(total_faith)
  total_com=np.array(total_com)
  total_suf=np.array(total_suf)

  if length==1:
      print(x.shape)
      xm=x[bz].reshape(1,1,112,112,112)
      print(y.shape)
      ym=y[bz].reshape(1,1)
      explanations=explanations.reshape(1,1,112,112,112)
      com,faith,suf=xai_score(xm,ym,explanations,device = device)
      min_com=com
      max_faith=faith
      ind_com=1
      ind_faith=1
  else:
      min_com=np.min(abs(total_com))
      max_faith=np.max(abs(total_faith))
      ind_com=np.argmin(abs(total_com))
      ind_faith=np.argmax(abs(total_faith))
  print(total_com,total_faith)
  print('MIN: ',min_com,' explaner: ',ind_com,' MAX: ',max_faith,' explainer: ',ind_faith)

  return min_com,max_faith
