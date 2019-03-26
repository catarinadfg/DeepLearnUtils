# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

from __future__ import division, print_function
from functools import partial,wraps
import numpy as np
from scipy.ndimage import shift,zoom,rotate, geometric_transform, map_coordinates
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import interp2d 

import trainutils


def applyOp(imgs,op,indices):
    indices=list(indices or range(len(imgs)))
    return tuple((op(im) if i in indices else im) for i,im in enumerate(imgs))
    

def randChoiceAugment(func):
    @wraps(func)
    def _func(*args,**kwargs):
        prob=kwargs.pop('prob',0.5)
        if not trainutils.randChoice(prob):
            return args
        else:
            return func(*args,**kwargs)
        
    return _func


def applyOpAugment(func):
    @wraps(func)
    def _func(*args,**kwargs):
        applyIndices=kwargs.pop('applyIndices',None)
        imgs,op=func(*args,**kwargs)
        return applyOp(imgs,op,applyIndices)
    
    return _func
        

@randChoiceAugment
@applyOpAugment
def transposeAugment(*imgs):
    '''Transpose each of `imgs'. If `applyIndices' is not None it must be the list of indices in `imgs' to augment.'''
#    if not trainutils.randChoice(prob): # `prob' chance of using this augment
#        return imgs
#    
#    op=partial(np.swapaxes,axis1=0,axis2=1)
#    return applyOp(imgs,op,applyIndices)
    
    return imgs,partial(np.swapaxes,axis1=0,axis2=1)


@randChoiceAugment
@applyOpAugment
def flipAugment(*imgs):
    '''
    Flip each of `imgs' with a random choice of up-down or left-right. If `applyIndices' is not None it must be the list 
    of indices in `imgs' to augment.
    '''
#    if not trainutils.randChoice(prob): # `prob' chance of using this augment
#        return imgs
#    
    op=np.fliplr if trainutils.randChoice() else np.flipud
#    return applyOp(imgs,op,applyIndices)
    return imgs,op


@randChoiceAugment
@applyOpAugment
def rot90Augment(*imgs):
    '''
    Rotate each of `imgs' a random choice of quarter, half, or three-quarter circle rotations. If `applyIndices' is not 
    None it must be the list of indices in `imgs' to augment.
    '''
#    if not trainutils.randChoice(prob): # `prob' chance of using this augment
#        return imgs
        
    op=partial(np.rot90,k=np.random.randint(1,3))
#    return applyOp(imgs,op,applyIndices)
    return imgs,op
        

@applyOpAugment
def normalizeAugment(*imgs):
    '''Normalize each of `imgs'. If `applyIndices' is not None it must be the list of indices in `imgs' to augment.'''
#    return applyOp(imgs,trainutils.rescaleArray,applyIndices)
    return imgs,trainutils.rescaleArray


@randChoiceAugment
@applyOpAugment
def randPatchAugment(*imgs,patchSize=(32,32), nonzeroIndex=-1):
    '''
    Randomly choose a patch from `imgs' of dimensions `patchSize'. The default probability of using this augment `prob'
    is 1.0. The patch will be chosen so that the image at index `nonzeroIndex' has non-zero pixels in it, this can be
    used to ensure the chosen patch includes segmented features not in the background. If `applyIndices' is not None it 
    must be the list of indices in `imgs' to augment.
    '''
#    if not trainutils.randChoice(prob): # `prob' chance of using this augment
#        return imgs
    
    testim=imgs[nonzeroIndex]
    h,w=testim.shape[:2]
    ph,pw=patchSize
    ry=np.random.randint(0,h-ph)
    rx=np.random.randint(0,w-pw)
    
    while testim[ry:ry+ph,rx:rx+pw].max()==0:
        ry=np.random.randint(0,h-ph)
        rx=np.random.randint(0,w-pw)
    
    op=lambda im: im[ry:ry+ph,rx:rx+pw]
    return imgs,op
#    return applyOp(imgs,op,applyIndices)


if __name__=='__main__':
    
    im=np.random.rand(128,128)
    
    imt=transposeAugment(im,prob=1.0)
    print(np.all(im.T==imt[0]))
    
    imf=flipAugment(im,prob=1.0)
    imr=rot90Augment(im,prob=1.0)
    
    im=np.random.rand(128,128)
    
    print(randPatchAugment(im)[0].shape)
    