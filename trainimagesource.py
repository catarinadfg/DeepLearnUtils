

from __future__ import division, print_function
import random

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift,zoom,rotate


def rescaleArray(arr,minv=0.0,maxv=1.0):
    '''Rescale the values of numpy array `arr' to the range `minv' to `maxv'.'''
    mina=np.min(arr)
    norm=(arr-mina)/(np.max(arr)-mina)
    return (norm*(maxv-minv))+minv


def comparePrediction(imgs,masks,logits,preds,title=''):
    for index in range(preds.shape[0]):
        im=imgs[index,...,0]
        logit=np.squeeze(logits[index])
        pred=np.squeeze(preds[index])
                
        padx=int(im.shape[0]-pred.shape[0])//2
        pady=int(im.shape[1]-pred.shape[1])//2
        if padx and pady:
            pred=np.pad(pred,((padx,padx),(pady,pady)),'constant',constant_values=np.min(pred))
            
        fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20,6))
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')
        ax[0].set_title(title)
        
        ax[1].imshow(logit)
        ax[2].imshow(pred)
        
        if masks is not None:
            ax[0].imshow(im+masks[index]*0.5)
            ax[3].imshow(np.stack([masks[index],pred*masks[index],pred],axis=2))
        else:
            ax[0].imshow(im)
            ax[3].imshow(im+pred*0.5)

        
def viewImages(img,mask):
    import pyqtgraph as pg
    img[...,0]=rescaleArray(img[...,0])
    mask=np.stack([mask*0.25]*3,axis=-1)
    pg.image(img+mask,xvals=np.arange(img.shape[0]+1))
    

def randChoice(prob=0.5):
    return random.random()<=prob


def imgBounds(img):
    '''Returns the minimum and maximum indices of non-zero lines in axis 0 of `img', followed by that for axis 1.'''
    ax0 = np.any(img, axis=0)
    ax1 = np.any(img, axis=1)
    return np.concatenate((np.where(ax0)[0][[0, -1]], np.where(ax1)[0][[0, -1]]))

    
def inBounds(x,y,margin,maxx,maxy):
    return x>=margin and y>=margin and x<(maxx-margin) and y<(maxy-margin)
    
    
def zeroMargins(img,margin):
    '''Returns True if the values within `margin' indices of the edges of `img' are 0.'''
    return not np.any(img[:,:margin]+img[:,-margin:]) and not np.any(img[:margin,:]+img[-margin:,:])


def shiftAugment(img,mask,margin=5):
    x,y=mask.shape
    mask1=mask.astype(np.int)
    shiftx=None
    shifty=None
    smask=None
    
    while smask is None or smask.max()==0 or not zeroMargins(smask,margin):
        shiftx=random.randint(-x/2,x/2)
        shifty=random.randint(-y/2,y/2)
        smask=shift(mask1,(shiftx,shifty))

    return shift(img,(shiftx,shifty)+tuple(0 for _ in range(2,img.ndim))),shift(mask,(shiftx,shifty))
    
    
def rotateAugment(img,mask,margin=5):
    angle=None
    mask1=mask.astype(np.int)
    
    # choose a new angle so long as the mask is in the margins
    while angle is None or not zeroMargins(rotate(mask1,angle,reshape=False),margin):
        angle=random.random()*360
    
    return rotate(img,angle,reshape=False),rotate(mask,angle,reshape=False)
    
    
def zoomAugment(img,mask,margin=5,zoomrange=0.5):
    def _copyzoom(im,zx,zy):
        temp=np.zeros_like(im)
        ztemp=zoom(im,(zx,zy)+tuple(1 for _ in range(2,im.ndim)),order=2)
        
        tx=temp.shape[0]//2
        ty=temp.shape[1]//2
        ztx=ztemp.shape[0]//2
        zty=ztemp.shape[1]//2
        wx=min(tx,ztx)
        wy=min(ty,zty)
        
        temp[tx-wx:tx+wx,ty-wy:ty+wy]=ztemp[ztx-wx:ztx+wx,zty-wy:zty+wy]
        return temp

    tempmask=None
    zx=None
    zy=None
    
    while tempmask is None or not zeroMargins(tempmask.astype(np.int),margin):
        zx=1.0+zoomrange-random.random()*zoomrange*2
        zy=1.0+zoomrange-random.random()*zoomrange*2
        tempmask=_copyzoom(mask,zx,zy)
        
    return _copyzoom(img,zx,zy),tempmask

    
def transposeAugment(img,mask):
    return np.swapaxes(img,0,1),mask.T


def flipAugment(img,mask):
    if randChoice():
        return np.fliplr(img),np.fliplr(mask)
    else:
        return np.flipud(img),np.flipud(mask)
        
        
def rot90Augment(img,mask):
    r=random.random()
    if r<0.33333:
        num=1
    elif r<0.66666:
        num=2
    else:
        num=3
        
    return np.rot90(img,num),np.rot90(mask,num)
    
    
class TrainImageSource(object):
    def __init__(self,images,masks,augments=[]):
        assert images.shape[1:3]==masks.shape[1:3],'%r != %r'%(images.shape[1:3],masks.shape[1:3])
        self.images=images
        self.masks=masks
        self.indices=range(self.images.shape[0])
        self.imgshape=list(self.images.shape)[1:3]
        self.augments=list(augments)
        self.channels=1 if self.images.ndim==3 else self.images.shape[3]
        
    def getBatch(self,numimgs):
        shape=[numimgs]+self.imgshape
        imgout=np.ndarray(shape+[self.channels],float)
        maskout=np.ndarray(shape+[1],float)
        
        for n in range(numimgs):
            randomindex=random.choice(self.indices)
            img=self.images[randomindex,...]
            mask=self.masks[randomindex,...]
            
            # apply each augment to the image and mask, giving each a 50% chance of being used
            for aug in self.augments:
                if randChoice():
                    img,mask=aug(img,mask)
                    
            imgout[n,...]=np.stack([img],axis=-1) if img.ndim==2 else img
            maskout[n,...,0]=mask
        
        return imgout,maskout
        

def getDefaultSource(imgs,masks):
    return TrainImageSource(imgs,masks,[zoomAugment,rotateAugment,rot90Augment,shiftAugment,transposeAugment,flipAugment])
        
    
if __name__=='__main__':
    (itrain,ivalid,ionline),(mtrain,mvalid,monline)=np.load('allimagemasks.npy')
    
    src=TrainImageSource(itrain,mtrain,[zoomAugment,rotateAugment,shiftAugment,rot90Augment,transposeAugment,flipAugment])
    
    img,mask=src.getBatch(10)
    #img,mask=ivalid,mvalid
    
    viewImages(img,np.squeeze(mask))
