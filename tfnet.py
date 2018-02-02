
from __future__ import division, print_function
import tensorflow as tf
from numpy import prod


conv2dtranspose=tf.layers.conv2d_transpose
conv2d=tf.layers.conv2d
batchnorm=tf.layers.batch_normalization
relu=tf.nn.relu
maxpool2d=tf.layers.max_pooling2d


def deconvUpsample2D(x,expandfactor=2,kernelsize=3,outchannels=None):
    '''
    Upsamples `x' for a factor of `expandfactor' using transpose convolutions. The convolution uses `expandfactor' as its
    stride value, `kernelsize' for size of kernel, and produces `outchannels' channels (filters) or the same number of
    channels as `x' if not given. 
    '''
    outchannels=outchannels or int(x.get_shape()[-1])
    return conv2dtranspose(x,outchannels,kernelsize,expandfactor,'same')


def upsampleConcat2D(x,concatx,expandfactor=2):
    '''Upsamples `x' by the given factor (stride) `expandfactor' then concat `concatx' to it.'''
    x=deconvUpsample2D(x,expandfactor)
    return tf.concat([concatx,x],-1)


def setChannels2D(x,outchannels,name=None):
    '''Returns a tensor converting `x' to have `outchannels' channels (filters).'''
    inchannels=int(x.get_shape()[-1])
    result=x
    
    if inchannels!=outchannels:
        result=conv2d(x,outchannels,1,name=name)
        
    return result


def residualUnit2D(x,outchannels,strides=1,kernelsize=3,numSubunits=1,isTraining=True):
    '''
    Generate a residual unit based on `x' with output having `outchannels' channels (filters). This is done by a sequence
    of batchnorm/relu/conv2d operations where the first convolution is done with strides given as `strides' and otherwise
    with 1. The sequence is done `numSubunits' number of times, thus if this is more than 1 and `strides' is greater than
    1 the first unit will downsample the output and then further units are applied to this. At the end of the sequence 
    the original value of `x' is added to the result, being itself downsampled using max pooling if needed.
    
    See: He, K. et al., "Identity Mappings in Deep Residual Networks", ECCV 2016.
    '''
    addx=x # original input is to be added to output from subunits
    convstrides=strides # use the given strides value only for the first subunit
    
    for su in range(numSubunits):
        with tf.variable_scope('ResSubunit'+str(su)):
            x=batchnorm(x,training=isTraining,fused=True)
            x=relu(x)
            x=conv2d(x,outchannels,kernelsize,convstrides,'same')
            convstrides=(1,1) # only allow first subunit to stride down
            
    with tf.variable_scope('ResAdd'):
        if prod(strides)!=1: # if x is strided down, apply max pooling to make addx match
            addx=maxpool2d(addx,strides,strides,'same')
    
        return x+setChannels2D(addx,outchannels) # add the original input tensor reshaped to match x


def unet2D(x,numClasses,channels,strides,kernelsize=3,numSubunits=1,isTraining=True):
    '''
    Generates a 2D unet network based on `x' to segment `numClasses' number of classes from input using residual units. 
    The `channels' and `strides' lists specify the number of channels (filters) each step will have and what the initial 
    stride is (1 for no reduction, 2 for 2x downsample) for each step. The `kernelsize' value specifies the convolution 
    kernel sizes, and `numSubunits' states how many subunit each residual unit will have. 
    
    See: Ronneberger, 0. et al., "U-Net: Convolutional Networks for Biomedical Image", MICCAI 2015
         Zhang, Z. et al., "Road Extraction by Deep Residual U-Net", IEEE Geoscience and Remote Sensing Letters
    '''
    assert len(channels)==len(strides)
    assert numClasses>0
    assert len(x.get_shape())==4

    # Decode channels, last channel set is for the bottom step so is omitted, final channel is number of classes so 
    # that the output of the decode stage will have as many channels as classes which is needed for making a prediction.
    dchannels=[numClasses]+list(channels[:-1])
    
    elist=[] # list of encode stages, this is build up in reverse order so that the decode stage works in reverse
    
    # encode stage
    for c,s,dc in zip(channels,strides,dchannels):
        i=len(elist)
        elist.insert(0,(i,x,dc,s))
            
        with tf.variable_scope('Encode%i'%i):
            x=residualUnit2D(x,c,s,kernelsize,numSubunits,isTraining)
            
    # decode stage
    for i,addx,c,s in elist:
        with tf.variable_scope('Decode%i'%i):
            x=upsampleConcat2D(x,addx,s)
            x=residualUnit2D(x,c,1,kernelsize,numSubunits,isTraining)
            
    # generate prediction outputs, x has shape (B,H,W,numClasses)
    with tf.variable_scope('Preds'):
        if numClasses==1:
            preds=tf.cast(x[...,0]>=0.5,tf.int32,'preds') # take probability
        else:
            preds=tf.argmax(x,3,'preds') # take index of most likely channel
            
    return x, preds
