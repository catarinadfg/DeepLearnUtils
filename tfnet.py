
from __future__ import division, print_function
import tensorflow as tf


conv2dtranspose=tf.layers.conv2d_transpose
conv2d=tf.layers.conv2d
batchnorm=tf.layers.batch_normalization
relu=tf.nn.relu
maxpool2d=tf.layers.max_pooling2d


def deconvUpsample2D(x,expandfactor=2,kernelsize=3,outchannels=None):
    outchannels=outchannels or int(x.get_shape()[-1])
    return conv2dtranspose(x,outchannels,kernelsize,expandfactor,'same')


def upsampleConcat2D(x,concatx,expandfactor=2):
    x=deconvUpsample2D(x,expandfactor)
    return tf.concat([concatx,x],-1)


def setChannels2D(x,outchannels,name=None):
    inchannels=int(x.get_shape()[-1])
    result=x
    
    if inchannels!=outchannels:
        result=conv2d(x,outchannels,1,name=name)
        
    return result


def residualUnit2D(x,outchannels,strides=1,kernelsize=3,numSubunits=1,isTraining=True):
    addx=x
    convstrides=strides
    
    for su in range(numSubunits):
        with tf.variable_scope('ResSubunit'+str(su)):
            x=batchnorm(x,training=isTraining,fused=True)
            x=relu(x)
            x=conv2d(x,outchannels,kernelsize,convstrides,'same')
            convstrides=(1,1) # only allow first subunit to stride down
            
    with tf.variable_scope('ResAdd'):
        if any(s!=1 for s in strides): # if x is strided down, apply max pooling to make addx match
            addx=maxpool2d(addx,strides,strides,'same')
    
        return x+setChannels2D(addx,outchannels)


def unet2D(x,numClasses,channels,strides,kernelsize=3,numSubunits=1,isTraining=True):
    assert len(channels)==len(strides)
    assert numClasses>0
    assert len(x.get_shape())==4

    dchannels=[numClasses]+list(channels[:-1])
    
    elist=[]
    
    for c,s,dc in zip(channels,strides,dchannels):
        i=len(elist)
        elist.insert(0,(i,x,dc,s))
            
        with tf.variable_scope('Encode%i'%i):
            x=residualUnit2D(x,c,s,kernelsize,numSubunits,isTraining)
            
    for i,addx,c,s in elist:
        with tf.variable_scope('Decode%i'%i):
            x=upsampleConcat2D(x,addx,s)
            x=residualUnit2D(x,c,1,kernelsize,numSubunits,isTraining)
            
    # generate prediction outputs
    with tf.variable_scope('Preds'):
        #x=setChannels2D(x,numClasses,'logits') # reduce `x' to have as many channels as classes
        
        if numClasses==1:
            preds=tf.cast(x[...,0]>=0.5,tf.int32,'preds') # take probability
        else:
            preds=tf.argmax(x,3,'preds')
            
    return x, preds
