

from __future__ import division, print_function
import random
import threading
import multiprocessing

import numpy as np
from scipy.ndimage import shift,zoom,rotate
    

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
    return np.any(img>img.min()) and not np.any(img[:,:margin]+img[:,-margin:]) and not np.any(img[:margin,:]+img[-margin:,:])


def shiftAugment(img,mask,margin=5):
    x,y=mask.shape[:2]
    mask1=mask.astype(np.int)
    shiftx=None
    shifty=None
    smask=None
    ishift0=tuple(0 for _ in range(2,img.ndim))
    mshift0=tuple(0 for _ in range(2,mask.ndim))
    
    while smask is None or smask.max()==0 or not zeroMargins(smask,margin):
        shiftx=random.randint(-x/2,x/2)
        shifty=random.randint(-y/2,y/2)
        smask=shift(mask1,(shiftx,shifty)+mshift0)

    return shift(img,(shiftx,shifty)+ishift0),shift(mask,(shiftx,shifty)+mshift0)
    
    
def rotateAugment(img,mask,margin=5):
    angle=None
    mask1=mask.astype(np.int)
    
    # choose a new angle so long as the mask is in the margins
    while angle is None or not zeroMargins(rotate(mask1,angle,reshape=False),margin):
        angle=random.random()*360
    
    return rotate(img,angle,reshape=False),rotate(mask,angle,reshape=False)
    
    
def zoomAugment(img,mask,margin=5,zoomrange=0.2):
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
        z=zoomrange-random.random()*zoomrange*2
        zx=z+1.0+zoomrange*0.25-random.random()*zoomrange*0.5
        zy=z+1.0+zoomrange*0.25-random.random()*zoomrange*0.5
        tempmask=_copyzoom(mask,zx,zy)
        
    return _copyzoom(img,zx,zy),tempmask

    
def transposeAugment(img,mask):
    return np.swapaxes(img,0,1),np.swapaxes(mask,0,1)


def flipAugment(img,out):
    if randChoice():
        return np.fliplr(img),np.fliplr(out)
    else:
        return np.flipud(img),np.flipud(out)
        
        
def rot90Augment(img,out):
    r=random.random()
    if r<0.33333:
        num=1
    elif r<0.66666:
        num=2
    else:
        num=3
        
    return np.rot90(img,num),np.rot90(out,num)
    
    
class TrainImageSource(object):
    def __init__(self,images,outputs,augments=[],numthreads=None,randomizeAugs=True):
        assert images.shape[:3]==outputs.shape[:3],'%r != %r'%(images.shape[:3],outputs.shape[:3])
        
        self.images=images
        self.outputs=outputs
        self.numthreads=numthreads
        self.randomizeAugs=randomizeAugs
        self.indices=list(range(self.images.shape[0]))
        #self.imgshape=list(self.images.shape)[1:]
        #self.outshape=list(self.outputs.shape)[1:]
        self.augments=list(augments)
        
    def getBatch(self,numimgs):
        imgtest,outtest=self._generateImagePair()
        imgs=np.ndarray((numimgs,)+imgtest.shape,imgtest.dtype)
        outs=np.ndarray((numimgs,)+outtest.shape,outtest.dtype)
        numthreads=min(numimgs,self.numthreads or multiprocessing.cpu_count())
        threads=[]

        def _generateForIndices(indices):
            for n in indices:
                imgs[n],outs[n]=self._generateImagePair()
                
        for indices in np.array_split(np.arange(numimgs),numthreads):
            t=threading.Thread(target=_generateForIndices,args=(indices,))
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
        
        return imgs,outs
    
    def _generateImagePair(self):
        randomindex=random.choice(self.indices)
        img=self.images[randomindex]
        out=self.outputs[randomindex]
        
        # apply each augment to the image and mask, giving each a 50% chance of being used if self.randomizeAugs is True
        for aug in self.augments:
            if not self.randomizeAugs or randChoice():
                img,out=aug(img,out)
                    
        return img,out
    