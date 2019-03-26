# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

from __future__ import division, print_function
from functools import partial,wraps
import numpy as np
from scipy.ndimage import shift,zoom,rotate,map_coordinates

import trainutils


def randChoiceAugment(func):
    '''
    Modify an augment to add a random choice parameter which chooses based on the keyword argument `prob' whether to
    apply the augment or just return the positional arguments. The default for `prob' is 0.5 (ie. coin-toss).
    '''
    @wraps(func)
    def _func(*args,**kwargs):
        prob=kwargs.pop('prob',0.5)
        
        if not trainutils.randChoice(prob):
            return args
        else:
            return func(*args,**kwargs)
        
    if _func.__doc__:
        _func.__doc__+='\n\nKeyword arg "prob": probability of applying this augment (default: 0.5)'
        
    return _func


def applyOpAugment(func):
    '''
    Modify an augment which is expected to return a callable that is then applied to each of the position arguments 
    indexed in keyword argument `applyIndices'. The return values from these applications are then returned. The default 
    for `applyIndices' is None meaning apply the callable to each argument.
    '''
    @wraps(func)
    def _func(*args,**kwargs):
        applyIndices=kwargs.pop('applyIndices',None)
        op=func(*args,**kwargs)
        indices=list(applyIndices or range(len(args)))
        
        return tuple((op(im) if i in indices else im) for i,im in enumerate(args))
    
    if _func.__doc__:
        _func.__doc__+='\n\nKeyword arg "applyIndices": indices of arrays to apply augment to (default: None meaning all)'
    
    return _func
        

@randChoiceAugment
@applyOpAugment
def transposeAugment(*arrs):
    '''Transpose axes 0 and 1 for each of `arrs'.'''
    return partial(np.swapaxes,axis1=0,axis2=1)


@randChoiceAugment
@applyOpAugment
def flipAugment(*arrs):
    '''Flip each of `arrs' with a random choice of up-down or left-right.'''
    return np.fliplr if trainutils.randChoice() else np.flipud


@randChoiceAugment
@applyOpAugment
def rot90Augment(*arrs):
    '''Rotate each of `arrs' a random choice of quarter, half, or three-quarter circle rotations.'''
    return partial(np.rot90,k=np.random.randint(1,3))
        

@applyOpAugment
def normalizeAugment(*arrs):
    '''Normalize each of `arrs'.'''
    return trainutils.rescaleArray


@applyOpAugment
def randPatchAugment(*arrs,patchSize=(32,32),maxcount=10, nonzeroIndex=-1):
    '''
    Randomly choose a patch from `arrs' of dimensions `patchSize'. if `nonzeroIndex' is not -1, the patch will be chosen 
    so that the image at index `nonzeroIndex' has positive non-zero pixels in it, this can be used to ensure the chosen 
    patch includes segmented features not in the background. 
    '''
    testim=arrs[nonzeroIndex]
    h,w=testim.shape[:2]
    ph,pw=patchSize
    ry=np.random.randint(0,h-ph)
    rx=np.random.randint(0,w-pw)
    
    if nonzeroIndex!=-1:
        for i in range(maxcount):
            if testim[ry:ry+ph,rx:rx+pw].max()>0:
                break
            
            ry=np.random.randint(0,h-ph)
            rx=np.random.randint(0,w-pw)

    return lambda im: im[ry:ry+ph,rx:rx+pw]


@randChoiceAugment
@applyOpAugment
def shiftAugment(*arrs,margin=5,dimfract=2,order=3,maxcount=10, nonzeroIndex=-1):
    '''Shift arrays randomly by `dimfract' fractions of the array dimensions.'''
    testim=arrs[nonzeroIndex]
    x,y=testim.shape[:2]
    seg=testim.astype(np.int)
    shiftx=np.random.randint(-x//dimfract,x//dimfract)
    shifty=np.random.randint(-y//dimfract,y//dimfract)
    
    def _shift(im):
        sval=(shiftx,shifty)+tuple(0 for _ in range(2,im.ndim))
        return shift(im,sval,order=order)
    
    if nonzeroIndex!=-1:
        for i in range(maxcount):
            seg=_shift(testim).astype(np.int32)
            if trainutils.zeroMargins(seg,margin):
                break
            
            shiftx=np.random.randint(-x//dimfract,x//dimfract)
            shifty=np.random.randint(-y//dimfract,y//dimfract)
            
    return _shift


@randChoiceAugment
@applyOpAugment
def rotateAugment(*arrs,margin=5,maxcount=10,nonzeroIndex=-1):
    '''Shift arrays randomly around the array center.'''
    
    angle=np.random.random()*360
    
    _rotate=partial(rotate,angle=angle,reshape=False)
    
    if nonzeroIndex!=-1:
        testim=arrs[nonzeroIndex]
        
        for i in range(maxcount):
            seg=_rotate(testim).astype(np.int32)
            if trainutils.zeroMargins(seg,margin):
                break
            
            angle=np.random.random()*360
        
    return _rotate


@randChoiceAugment
@applyOpAugment
def zoomAugment(*arrs,margin=5,zoomrange=0.2,maxcount=10,nonzeroIndex=-1):
    '''Return the image/mask pair zoomed by a random amount with the mask kept within `margin' pixels of the edges.'''
    
    z=zoomrange-np.random.random()*zoomrange*2
    zx=z+1.0+zoomrange*0.25-np.random.random()*zoomrange*0.5
    zy=z+1.0+zoomrange*0.25-np.random.random()*zoomrange*0.5
        
    def _zoom(im):
        ztemp=zoom(im,(zx,zy)+tuple(1 for _ in range(2,im.ndim)),order=2)
        return trainutils.resizeCenter(ztemp,*im.shape)
    
    if nonzeroIndex!=-1:
        testim=arrs[nonzeroIndex]
        
        for i in range(maxcount):
            seg=_zoom(testim).astype(np.int32)
            if trainutils.zeroMargins(seg,margin):
                break
            
            z=zoomrange-np.random.random()*zoomrange*2
            zx=z+1.0+zoomrange*0.25-np.random.random()*zoomrange*0.5
            zy=z+1.0+zoomrange*0.25-np.random.random()*zoomrange*0.5
            
    return _zoom

  
@randChoiceAugment
@applyOpAugment
def deformAugmentPIL(*arrs,defrange=25,numControls=3,margin=2):
    '''Deforms arrays randomly with a deformation grid of size `numControls'**2 with `margins' grid values fixed.'''
    from PIL import Image
    
    h,w = arrs[0].shape[:2]
    
    imshift=np.zeros((2,numControls+margin*2,numControls+margin*2))
    imshift[:,margin:-margin,margin:-margin]=np.random.randint(-defrange,defrange,(2,numControls,numControls))

    imshiftx=np.array(Image.fromarray(imshift[0]).resize((w,h),Image.QUAD))
    imshifty=np.array(Image.fromarray(imshift[1]).resize((w,h),Image.QUAD))
        
    y,x=np.meshgrid(np.arange(w), np.arange(h))
    indices=np.reshape(x+imshiftx, (-1, 1)),np.reshape(y+imshifty, (-1, 1))

    def _mapChannels(im):
        if len(im.shape)==2:
            result=map_coordinates(im,indices, order=1, mode='constant')
        else:
            result=np.concatenate([_mapChannels(im[...,i]) for i in range(im.shape[-1])])
            
        return result.reshape(im.shape)
    
    return _mapChannels



if __name__=='__main__':
    
    im=np.random.rand(128,128)
    
    imt=transposeAugment(im,prob=1.0)
    print(np.all(im.T==imt[0]))
    
    imf=flipAugment(im,prob=1.0)
    imr=rot90Augment(im,prob=1.0)
    
    im=np.random.rand(128,128)
    
    print(randPatchAugment(im,patchSize=(30,34))[0].shape)
    
    print(shiftAugment(im,prob=1.0)[0].shape)
    
    print(rotateAugment(im,prob=1.0)[0].shape)
    
    print(zoomAugment(im,prob=1.0)[0].shape)
    
    print(deformAugmentPIL(im,prob=1.0)[0].shape)
    