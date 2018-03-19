

from __future__ import division, print_function
import random
import threading
import multiprocessing

import numpy as np
from scipy.ndimage import shift,zoom,rotate
from scipy.ndimage.filters import gaussian_filter

from trainutils import rescaleArray
    
try:
    import queue
except:
    import Queue as queue
    

def randChoice(prob=0.5):
    return random.random()<=prob


def imgBounds(img):
    '''Returns the minimum and maximum indices of non-zero lines in axis 0 of `img', followed by that for axis 1.'''
    ax0 = np.any(img, axis=0)
    ax1 = np.any(img, axis=1)
    return np.concatenate((np.where(ax0)[0][[0, -1]], np.where(ax1)[0][[0, -1]]))

    
def inBounds(x,y,margin,maxx,maxy):
    '''Returns True if (x,y) is within the rectangle (margin,margin,maxx-margin,maxy-margin).'''
    #return x>=margin and y>=margin and x<(maxx-margin) and y<(maxy-margin)
    return margin<=x<(maxx-margin) and margin<y<(maxy-margin)
    
    
def zeroMargins(img,margin):
    '''Returns True if the values within `margin' indices of the edges of `img' are 0.'''
    return img.max()>img.min() and not np.any(img[:,:margin]+img[:,-margin:]) and not np.any(img[:margin,:]+img[-margin:,:])


def shiftAugment(img,mask,margin=5,prob=0.5):
    if randChoice(prob): # `prob' chance of using this augment
        return img,mask
    
    x,y=mask.shape[:2]
    mask1=mask.astype(np.int)
    shiftx=None
    shifty=None
    smask=None
    ishift0=tuple(0 for _ in range(2,img.ndim))
    mshift0=tuple(0 for _ in range(2,mask.ndim))
    
    while smask is None or not zeroMargins(smask,margin):
        shiftx=random.randint(-x/2,x/2)
        shifty=random.randint(-y/2,y/2)
        smask=shift(mask1,(shiftx,shifty)+mshift0)

    return shift(img,(shiftx,shifty)+ishift0),shift(mask,(shiftx,shifty)+mshift0)
    
    
def rotateAugment(img,mask,margin=5,prob=0.5):
    if randChoice(prob): # `prob' chance of using this augment
        return img,mask
    
    angle=None
    mask1=mask.astype(np.int)
    
    # choose a new angle so long as the mask is in the margins
    while angle is None or not zeroMargins(rotate(mask1,angle,reshape=False),margin):
        angle=random.random()*360
    
    return rotate(img,angle,reshape=False),rotate(mask,angle,reshape=False)
    
    
def zoomAugment(img,mask,margin=5,zoomrange=0.2,prob=0.5):
    if randChoice(prob): # `prob' chance of using this augment
        return img,mask
    
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

    
def transposeAugment(img,mask,prob=0.5):
    if randChoice(prob): # `prob' chance of using this augment
        return img,mask
    

    return np.swapaxes(img,0,1),np.swapaxes(mask,0,1)


def flipAugment(img,out,prob=0.5):
    if randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    if randChoice():
        return np.fliplr(img),np.fliplr(out)
    else:
        return np.flipud(img),np.flipud(out)
        
        
def rot90Augment(img,out,prob=0.5):
    if randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    r=random.random()
    if r<0.33333:
        num=1
    elif r<0.66666:
        num=2
    else:
        num=3
        
    return np.rot90(img,num),np.rot90(out,num)


def randomCrop(img,out,box=(64,64)):
    h,w=box
    h2=h//2
    w2=w//2
    x=None
    y=None
    
    while x==None or np.sum(img[y-h2:y+h2,x-w2:x+w2])==0:
        x=w2+int(random.random()*(img.shape[1]-w-1))
        y=h2+int(random.random()*(img.shape[0]-h-1))
    
    return img[y-h2:y+h2,x-w2:x+w2],out[y-h2:y+h2,x-w2:x+w2]
    

def blurOut(img,out,sigma=2):
    return img,gaussian_filter(out,sigma)
    

def normalizeBoth(img,out):
    return rescaleArray(img),rescaleArray(out)

    
class TrainImageSource(object):
    def __init__(self,images,outputs,augments=[],numthreads=None):
        assert images.shape[:3]==outputs.shape[:3],'%r != %r'%(images.shape[:3],outputs.shape[:3])
        
        self.images=images
        self.outputs=outputs
        self.numthreads=numthreads
        self.indices=list(range(self.images.shape[0]))
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
            img,out=aug(img,out)
                    
        return img,out
    
    def getAsyncGenerator(self,numimgs,queueLength=1):
        batches=queue.Queue(queueLength)
        
        def _batchThread():
            while True:
                    batches.put(self.getBatch(numimgs))
                
        batchthread=threading.Thread(target=_batchThread)
        batchthread.daemon=True
        batchthread.start()
        
        def _dequeue():
            return batches.get()
                
        return _dequeue
    