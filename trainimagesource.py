

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
    '''Returns True if a randomly chosen number is less than or equal to `prob', by default this is a 50/50 chance.'''
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


def shiftMaskAugment(img,mask,margin=5,prob=0.5):
    '''Return the image/mask pair shifted by a random amount with the mask kept within `margin' pixels of the edges.'''
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
    
    
def rotateMaskAugment(img,mask,margin=5,prob=0.5):
    '''Return the image/mask pair rotated by a random amount with the mask kept within `margin' pixels of the edges.'''
    if randChoice(prob): # `prob' chance of using this augment
        return img,mask
    
    angle=None
    mask1=mask.astype(np.int)
    
    # choose a new angle so long as the mask is in the margins
    while angle is None or not zeroMargins(rotate(mask1,angle,reshape=False),margin):
        angle=random.random()*360
    
    return rotate(img,angle,reshape=False),rotate(mask,angle,reshape=False)
    
    
def zoomMaskAugment(img,mask,margin=5,zoomrange=0.2,prob=0.5):
    '''Return the image/mask pair zoomed by a random amount with the mask kept within `margin' pixels of the edges.'''
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

    
def transposeBothAugment(img,out,prob=0.5):
    '''Transpose both inputs.'''
    if randChoice(prob): # `prob' chance of using this augment
        return img,out

    return np.swapaxes(img,0,1),np.swapaxes(out,0,1)


def flipBothAugment(img,out,prob=0.5):
    '''Flip both inputs with a random choice of up-down or left-right.'''
    if randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    if randChoice():
        return np.fliplr(img),np.fliplr(out)
    else:
        return np.flipud(img),np.flipud(out)
        
        
def rot90BothAugment(img,out,prob=0.5):
    '''Rotate both inputs a random choice of quarter, half, or three-quarter circle rotations.'''
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


def normalizeBothAugment(img,out):
    '''Normalize both inputs.'''
    return rescaleArray(img),rescaleArray(out)


def randomCropBothAugment(img,out,box=(64,64)):
    '''Randomly crop both inputs to return image pairs of size `box'.'''
    h,w=box
    h2=h//2
    w2=w//2
    x=None
    y=None
    
    while x==None or np.sum(img[y-h2:y+h2,x-w2:x+w2])==0:
        x=w2+int(random.random()*(img.shape[1]-w-1))
        y=h2+int(random.random()*(img.shape[0]-h-1))
    
    return img[y-h2:y+h2,x-w2:x+w2],out[y-h2:y+h2,x-w2:x+w2]


def normalizeImageAugment(img,out):
    '''Normalize image input.'''
    return rescaleArray(img),out


def shiftImgAugment(img,out,prob=0.5,dimfract=2):
    '''Shift `img' by a random amount and leave `out' unchanged.'''
    if randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    try:
        y,x=img.shape[:2]
        shifty=random.randint(-y//dimfract,y//dimfract)
        shiftx=random.randint(-x//dimfract,x//dimfract)
    except:
        print(y,x)
        raise
    ishift0=tuple(0 for _ in range(2,img.ndim))
    
    return shift(img,(shifty,shiftx)+ishift0),out
    
    
def rotateImgAugment(img,out,prob=0.5):
    '''Rotate `img' by a random amount and leave `out' unchanged.'''
    if randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    angle=random.random()*360
    
    return rotate(img,angle,reshape=False),out
    
    
def zoomImgAugment(img,out,zoomrange=0.2,prob=0.5):
    '''Zoom `img' by a random amount and leave `out' unchanged.'''
    if randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    z=zoomrange-random.random()*zoomrange*2
    zx=z+1.0+zoomrange*0.25-random.random()*zoomrange*0.5
    zy=z+1.0+zoomrange*0.25-random.random()*zoomrange*0.5
    
    temp=np.zeros_like(img)
    ztemp=zoom(img,(zx,zy)+tuple(1 for _ in range(2,img.ndim)),order=2)
    
    tx=temp.shape[0]//2
    ty=temp.shape[1]//2
    ztx=ztemp.shape[0]//2
    zty=ztemp.shape[1]//2
    wx=min(tx,ztx)
    wy=min(ty,zty)
    
    temp[tx-wx:tx+wx,ty-wy:ty+wy]=ztemp[ztx-wx:ztx+wx,zty-wy:zty+wy]
    
    return temp,out


def transposeImgAugment(img,out,prob=0.5):
    '''Transpose `img' and leave `out' unchanged.'''
    if randChoice(prob): # `prob' chance of using this augment
        return img,out

    return np.swapaxes(img,0,1),out


def flipImgAugment(img,out,prob=0.5):
    '''Flip `img' with a random choice of up-down or left-right and leave `out' unchanged.'''
    if randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    if randChoice():
        return np.fliplr(img),out
    else:
        return np.flipud(img),out
        
        
def rot90ImgAugment(img,out,prob=0.5):
    '''Rotate `img' a random choice of quarter, half, or three-quarter circle rotations, and leave `out' unchanged.'''
    if randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    r=random.random()
    if r<0.33333:
        num=1
    elif r<0.66666:
        num=2
    else:
        num=3
        
    return np.rot90(img,num),out
    

def blurImgAugment(img,out,sigma=2):
    '''Blur `img' with gaussian filter using `sigma' with out unchanged.'''
    return img,gaussian_filter(out,sigma)

    
class TrainImageSource(object):
    '''
    Given a set of images and outputs associated with them, this defines an infinite queue of minibatches with image
    augmentations applied to either the image or both data elements.
    '''
    def __init__(self,images,outputs,augments=[],numthreads=None):
        '''
        Initialize the queue with `images' as the image list and `outputs' as the list of associated per-image out values.
        The `images' array is expected to be in BHWC or BDHWC index ordering.
        '''
        assert images.shape[0]==outputs.shape[0],'%r != %r'%(images.shape[0],outputs.shape[0])
        
        self.images=images
        self.outputs=outputs
        self.numthreads=numthreads
        self.indices=list(range(self.images.shape[0]))
        self.probs=np.ones((self.images.shape[0],),np.float32)
        self.augments=list(augments)
        self.batchthread=None
        self.batchqueue=None
        
    def getBatch(self,numimgs):
        imgtest,outtest=self._generateImagePair()
        imgs=np.ndarray((numimgs,)+imgtest.shape,imgtest.dtype)
        outs=np.ndarray((numimgs,)+outtest.shape,outtest.dtype)
        numthreads=min(numimgs,self.numthreads or multiprocessing.cpu_count())
        threads=[]
        
        chosenindices=np.random.choice(self.indices,numimgs,p=self.probs/self.probs.sum())
        
        indexpairs=np.stack([np.arange(numimgs),chosenindices],axis=1)

        def _generateForIndices(indices):
            for n,c in indices:
                imgs[n],outs[n]=self._generateImagePair(c)
                
        for indices in np.array_split(indexpairs,numthreads):
            t=threading.Thread(target=_generateForIndices,args=(indices,))
            t.start()
            threads.append(t)
            
        for t in threads:
            t.join()
        
        return imgs,outs
    
    def _generateImagePair(self,index=None):
        randomindex=index if index is not None else random.choice(self.indices)
        img=self.images[randomindex]
        out=self.outputs[randomindex]
        
        # apply each augment to the image and mask, giving each a 50% chance of being used if self.randomizeAugs is True
        for aug in self.augments:
            img,out=aug(img,out)
                    
        return img,out
    
    def getAsyncFunc(self,numimgs,queueLength=1):
        if self.batchqueue is None:
            self.batchqueue=queue.Queue(queueLength)
            
            def _batchThread():
                while True:
                        self.batchqueue.put(self.getBatch(numimgs))
                    
            self.batchthread=threading.Thread(target=_batchThread)
            self.batchthread.daemon=True
            self.batchthread.start()
        
        return self.batchqueue.get
    
    