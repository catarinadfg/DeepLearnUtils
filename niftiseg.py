
from __future__ import division, print_function
import argparse
import nibabel
import tensorflow as tf
import numpy as np

from segmenter import Segmenter
from trainimagesource import rescaleArray


def applySegmentation(metafilename,img,device='/gpu:0',conf=None):
    '''
    Loads the graph from the meta file `metafilename' and predicts segmentations for the image stack `img'. The graph is
    expected to have a collection "endpoints" storing the x,y_,y,ypred list of tensors where x is the input and ypred the
    predicted segmentation. The first 2 dimensions of `img' must be the XY dimensions, other dimensions are flattened out.
    The values of `img' are also expected to be normalized. Returns a stack of binary masks with the same shape as `img'.
    '''
    seg=Segmenter(metafilename,device,conf)
    
    origshape=tuple(img.shape)
    tf.logging.info('Input dimensions: %r'%(origshape,))
    
    shape=origshape+(1,1,1) # add extra dimensions to the shape to make a 4D shape
    shape=shape[:5] # clip extra dimensions off so that this is a 5D shape description
    width,height,slices,timesteps,depth=shape
    
    img=img.astype(np.dtype('<f8')).reshape(shape) # convert input into a 5D image of shape XYZTC
    img=rescaleArray(img)
    
    for s in range(slices):
        tf.logging.info('Segmenting slice %s'%s)
        for t in range(timesteps):
            st=img[...,s,t,:]
            
            if st.max()>st.min(): # segment if the image is not blank
                pred=seg.apply(st)
                img[...,s,t,:]=0
                img[...,s,t,0]=pred
                
    return img.reshape(origshape) 
            

if __name__=='__main__':
    parser=argparse.ArgumentParser('niftiseg.py')
    parser.add_argument('metafilename',help='Path to a Tensorflow meta graph file')
    parser.add_argument('infile',help='Nifti file to segment')
    parser.add_argument('outfile',help='Nifti output filename')
    parser.add_argument('--device',help='Tensorflow device name to compute on',default='/gpu:0')
    args=parser.parse_args()
    
    tf.logging.set_verbosity(tf.logging.INFO)
    
    tf.logging.info('Loading '+args.infile)
    infile=nibabel.load(args.infile)
    dat=infile.get_data()
    
    preds=applySegmentation(args.metafilename,dat,args.device)
    
    tf.logging.info('Saving '+args.outfile)
    outfile = nibabel.Nifti1Image(preds.astype(dat.dtype), infile.affine, infile.header)
    nibabel.save(outfile,args.outfile)
    
