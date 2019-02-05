# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file



from __future__ import division, print_function
import random
import threading

import numpy as np
from scipy.ndimage import shift,zoom,rotate, geometric_transform, map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp2d 

from datasource import DataSource
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
    return margin<=x<(maxx-margin) and margin<=y<(maxy-margin)
    
    
def zeroMargins(img,margin):
    '''Returns True if the values within `margin' indices of the edges of `img' are 0.'''
    return img.max()>img.min() and not np.any(img[:,:margin]+img[:,-margin:]) and not np.any(img[:margin,:]+img[-margin:,:])


def shiftMaskAugment(img,mask,margin=5,prob=0.5,dimfract=2,order=3,maxcount=10):
    '''Return the image/mask pair shifted by a random amount with the mask kept within `margin' pixels of the edges.'''
    if not randChoice(prob): # `prob' chance of using this augment
        return img,mask
    
    x,y=mask.shape[:2]
    mask1=mask.astype(np.int)
    shiftx=None
    shifty=None
    smask=None
    ishift0=tuple(0 for _ in range(2,img.ndim))
    mshift0=tuple(0 for _ in range(2,mask.ndim))
    
    while smask is None or not zeroMargins(smask,margin):
        shiftx=random.randint(-x//dimfract,x//dimfract)
        shifty=random.randint(-y//dimfract,y//dimfract)
        smask=shift(mask1,(shiftx,shifty)+mshift0,order=order)
        
        maxcount-=1
        if maxcount<=0:
            return img,mask

    return shift(img,(shiftx,shifty)+ishift0,order=order),shift(mask,(shiftx,shifty)+mshift0,order=order)
    
    
def rotateMaskAugment(img,mask,margin=5,prob=0.5,maxcount=10):
    '''Return the image/mask pair rotated by a random amount with the mask kept within `margin' pixels of the edges.'''
    if not randChoice(prob): # `prob' chance of using this augment
        return img,mask
    
    angle=None
    mask1=mask.astype(np.int)
    
    # choose a new angle so long as the mask is in the margins
    while angle is None or not zeroMargins(rotate(mask1,angle,reshape=False),margin):
        angle=random.random()*360
        
        maxcount-=1
        if maxcount<=0:
            return img,mask
    
    return rotate(img,angle,reshape=False),rotate(mask,angle,reshape=False)
    
    
def zoomMaskAugment(img,mask,margin=5,zoomrange=0.2,prob=0.5,maxcount=10):
    '''Return the image/mask pair zoomed by a random amount with the mask kept within `margin' pixels of the edges.'''
    if not randChoice(prob): # `prob' chance of using this augment
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
        
        maxcount-=1
        if maxcount<=0:
            return img,mask
        
    return _copyzoom(img,zx,zy),tempmask


def _mapImageChannels(image,indices):
    if len(image.shape)==2:
        result=map_coordinates(image,indices, order=1, mode='constant').reshape(image.shape)
    else:
        result=np.concatenate([_mapImageChannels(image[...,i],indices) for i in range(image.shape[-1])])
        
    return result.reshape(image.shape)
    
    
def deformBothAugmentPIL(image,seg,defrange=25,numControls=3,margin=2,prob=0.5):
    from PIL import Image
    
    if not randChoice(prob): # `prob' chance of using this augment
        return image,seg
    
    h,w = image.shape[:2]
    
    imshift=np.zeros((2,numControls+margin*2,numControls+margin*2))
    imshift[:,margin:-margin,margin:-margin]=np.random.randint(-defrange,defrange,(2,numControls,numControls))

    imshiftx=np.array(Image.fromarray(imshift[0]).resize((w,h),Image.QUAD))
    imshifty=np.array(Image.fromarray(imshift[1]).resize((w,h),Image.QUAD))
        
    y,x=np.meshgrid(np.arange(w), np.arange(h))
    indices=np.reshape(x+imshiftx, (-1, 1)),np.reshape(y+imshifty, (-1, 1))

    imagedef=_mapImageChannels(image,indices)
    segdef=_mapImageChannels(seg,indices)
    return imagedef,segdef


def deformImgAugmentPIL(image,out,defrange=25,numControls=3,margin=2,prob=0.5):
    from PIL import Image
    
    if not randChoice(prob): # `prob' chance of using this augment
        return image,out
    
    h,w = image.shape[:2]
    
    imshift=np.zeros((2,numControls+margin*2,numControls+margin*2))
    imshift[:,margin:-margin,margin:-margin]=np.random.randint(-defrange,defrange,(2,numControls,numControls))

    imshiftx=np.array(Image.fromarray(imshift[0]).resize((w,h),Image.QUAD))
    imshifty=np.array(Image.fromarray(imshift[1]).resize((w,h),Image.QUAD))
        
    y,x=np.meshgrid(np.arange(w), np.arange(h))
    indices=np.reshape(x+imshiftx, (-1, 1)),np.reshape(y+imshifty, (-1, 1))

    imagedef=_mapImageChannels(image,indices)
    return imagedef,out


def _mapping(coords,interx,intery):
    y,x=coords[:2]
    dx=interx(x,y)
    dy=intery(x,y)

    return (y+dy,x+dx)+tuple(coords[2:])


def deformBothAugment(img,mask,defrange=5,prob=0.5):
    '''
    Deform the central parts of the image/mask pair using interpolated randomized deformation. This defines a 4x4 grid
    over the image, perturbs the central 4 points by a random multiple of `defrange' in either direction for both 
    dimensions, and then transforms the image and mask based on this deformation field. 
    '''
    if not randChoice(prob): # `prob' chance of using this augment
        return img,mask
    
    grid=np.zeros((4,4,2),int)
    y=np.linspace(0,img.shape[0],grid.shape[0])
    x=np.linspace(0,img.shape[1],grid.shape[1])
    
    grid[1:3,1:3,:]=2*defrange*np.random.random_sample((2,2,2))-defrange
        
    interx=interp2d(x,y,grid[...,0],'linear')
    intery=interp2d(x,y,grid[...,1],'linear')
    xargs=(interx,intery)
        
    iout=geometric_transform(img,_mapping,extra_arguments=xargs)
    mout=geometric_transform(mask,_mapping,extra_arguments=xargs)
    return iout,mout

    
def transposeBothAugment(img,out,prob=0.5):
    '''Transpose both inputs.'''
    if not randChoice(prob): # `prob' chance of using this augment
        return img,out

    return np.swapaxes(img,0,1),np.swapaxes(out,0,1)


def flipBothAugment(img,out,prob=0.5):
    '''Flip both inputs with a random choice of up-down or left-right.'''
    if not randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    if randChoice():
        return np.fliplr(img),np.fliplr(out)
    else:
        return np.flipud(img),np.flipud(out)
        
        
def rot90BothAugment(img,out,prob=0.5):
    '''Rotate both inputs a random choice of quarter, half, or three-quarter circle rotations.'''
    if not randChoice(prob): # `prob' chance of using this augment
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


def shiftImgAugment(img,out,prob=0.5,dimfract=2,order=3):
    '''Shift `img' by a random amount and leave `out' unchanged.'''
    if not randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    try:
        y,x=img.shape[:2]
        shifty=random.randint(-y//dimfract,y//dimfract)
        shiftx=random.randint(-x//dimfract,x//dimfract)
    except:
        print(y,x)
        raise
    ishift0=tuple(0 for _ in range(2,img.ndim))
    
    return shift(img,(shifty,shiftx)+ishift0,order=order),out
    
    
def rotateImgAugment(img,out,prob=0.5):
    '''Rotate `img' by a random amount and leave `out' unchanged.'''
    if not randChoice(prob): # `prob' chance of using this augment
        return img,out
    
    angle=random.random()*360
    
    return rotate(img,angle,reshape=False),out
    
    
def zoomImgAugment(img,out,zoomrange=0.2,prob=0.5):
    '''Zoom `img' by a random amount and leave `out' unchanged.'''
    if not randChoice(prob): # `prob' chance of using this augment
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
    if not randChoice(prob): # `prob' chance of using this augment
        return img,out

    return np.swapaxes(img,0,1),out


def flipImgAugment(img,out,prob=0.5):
    '''Flip `img' with a random choice of up-down or left-right and leave `out' unchanged.'''
    if not randChoice(prob): # `prob' chance of using this augment
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


class TrainImageSource(DataSource):
    '''
    Given a set of images and outputs associated with them, this defines an infinite queue of minibatches with image
    augmentations applied to either the image or both data elements.
    '''
    def __init__(self,images,outputs,augments=[],numthreads=None):
        DataSource.__init__(self,images,outputs,augments=augments,selectProbs=np.ones((images.shape[0],),np.float32)/images.shape[0])
        self.batchthread=None
        self.batchqueue=None
        
    def getBatch(self,numimgs):
        with self.threadBatchGen(numimgs) as gen:
            return gen()
        
    def getAsyncFunc(self,numimgs,queueLength=1):
        if self.batchqueue is None:
            self.batchqueue=queue.Queue(queueLength)
            
            def _batchThread():
                with self.threadBatchGen(numimgs) as gen:
                    while True:
                        self.batchqueue.put(gen())
                    
            self.batchthread=threading.Thread(target=_batchThread)
            self.batchthread.daemon=True
            self.batchthread.start()
        
        return self.batchqueue.get
        
