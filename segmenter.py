
import os

import numpy as np
from scipy import ndimage

import tensorflow as tf

from trainutils import rescaleArray


class Segmenter(object):
    defaultconf = tf.ConfigProto(allow_soft_placement=True)
    defaultconf.gpu_options.allow_growth = True
    
    def __init__(self,metafilename,device='/gpu:0',conf=None):
        with tf.device(device):
            self.device=device
            conf=conf or self.defaultconf
            
            self.graph=tf.Graph()
            self.sess = tf.InteractiveSession(config=conf,graph=self.graph)
            
            with self.graph.as_default():
                tf.logging.info('Importing graph '+metafilename)
                saver = tf.train.import_meta_graph(metafilename,clear_devices=True)
            
                tf.logging.info('Restoring checkpoint')
                saver.restore(self.sess, os.path.splitext(metafilename)[0])
                
                tf.logging.info('Retrieving endpoints')
                self.x,_,self.y,self.ypred=tf.get_collection('endpoints')[:4] 
                
                xshape=self.x.get_shape().as_list() # (BDHWC) or (BHWC)
                
                self.xwidth,self.xheight,self.xchannels=xshape[2 if len(xshape)==5 else 1:] # image dimensions
                    
                tf.logging.info('Graph input dimensions: %r'%((self.xwidth,self.xheight,self.xchannels),))
            
                self.xw2=self.xwidth//2
                self.xh2=self.xheight//2
                
                self.tempimg=np.ndarray(((1,1) if len(xshape)==5 else (1,))+(self.xwidth,self.xheight,self.xchannels))
                self.feeddict={self.x:self.tempimg}
            
    def apply(self,img, keepLargest=True,normalizeImg=True,resultScale=None):
        assert img.ndim in (2,3), 'Image dimension must be 2 or 3, is %r'%img.ndim
    
        if img.ndim==2: # extend a (H,W) stack to be (H,W,C) with a single channel
            img=np.expand_dims(img,axis=-1)
            assert self.xchannels==1,'Input image requires %i channels'%self.xchannels
            
        assert img.shape[-1]==self.xchannels, 'Input image channels %r does not match network input channels %r'%(img.shape[-1],self.xchannels)
        
        result=np.zeros(img.shape[:-1],np.float32)
        imin=img.min()
        imax=img.max()
        
        if normalizeImg:
            tf.logging.info('Input range: %r %r'%(img.min(),img.max()))
            img=rescaleArray(img)
        
        if imax>imin:
            tf.logging.info('Segmenting image of dimensions %r on device %r'%(img.shape,self.device))
            #img=(img-imin)/(imax-imin) # normalize image
            width,height,depth=img.shape
        
            w2=width//2
            h2=height//2
            wmin=min(w2,self.xw2)
            hmin=min(h2,self.xh2)
            
            st=img[w2-wmin:w2+wmin,h2-hmin:h2+hmin]
            self.tempimg[...,self.xw2-wmin:self.xw2+wmin,self.xh2-hmin:self.xh2+hmin,:]=st
        
            pred=self.sess.run(self.ypred,feed_dict=self.feeddict)
            pred=np.squeeze(pred[0]) # pred[0,0] if pred.ndim==4 else pred[0]
            
            labeled,numfeatures=ndimage.label(pred) # label each separate object with a different number
            if keepLargest and numfeatures>1: # if there's more than one object in the segmentation, keep only the largest as the best guess
                tf.logging.info('Segment size: %r'%np.sum(pred))    
                tf.logging.info('Isolating largest segment feature')
                sums=ndimage.sum(pred,labeled,range(numfeatures+1)) # sum the pixels under each label
                maxfeature=np.where(sums==max(sums)) # choose the maximum sum whose index will be the label number
                pred=pred*(labeled==maxfeature) # mask out the prediction under the largest label
            
            result[w2-wmin:w2+wmin,h2-hmin:h2+hmin]=pred[self.xw2-wmin:self.xw2+wmin,self.xh2-hmin:self.xh2+hmin]
            
        if resultScale is not None:
            tf.logging.info('Result range: %f %f'%(result.min(),result.max()))
            result=rescaleArray(result,0,float(resultScale))
            tf.logging.info('Result range: %f %f'%(result.min(),result.max()))
            
        return result
    