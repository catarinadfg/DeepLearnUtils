# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

from __future__ import print_function,division
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def samePadding(kernelsize):
    if isinstance(kernelsize,tuple):
        return tuple((k-1)//2 for k in kernelsize)
    else:
        return (kernelsize-1)//2
    

class BinaryDiceLoss(_Loss):
    def forward(self, source, target, smooth=1e-5):
        batchsize = target.size(0)
        probs = source.float().sigmoid()
        psum = probs.view(batchsize, -1)
        tsum = target.float().view(batchsize, -1)
        
        intersection=psum*tsum
        sums=psum+tsum

        score = 2.0 * (intersection.sum(1) + smooth) / (sums.sum(1) + smooth)
        score = 1 - score.sum() / batchsize
        return score
    

class Convolution2D(nn.Module):
    def __init__(self,inChannels,outChannels,strides=1,kernelsize=3,normalizeFunc=None):
        super(Convolution2D,self).__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        padding=samePadding(kernelsize)
        normalizeFunc=normalizeFunc or nn.InstanceNorm2d #if useInstanceNorm else nn.BatchNorm2d
        
        self.conv=nn.Sequential(
            nn.Conv2d(inChannels,outChannels,kernel_size=kernelsize,stride=strides,padding=padding),
            normalizeFunc(outChannels),#,track_running_stats=True), 
            nn.modules.PReLU()
        )
        
    def forward(self,x):
        return self.conv(x)
        

class ResidualUnit2D(nn.Module):
    def __init__(self, inChannels,outChannels,strides=1,kernelsize=3,subunits=2,normalizeFunc=None):
        super(ResidualUnit2D,self).__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        
        padding=samePadding(kernelsize)
        seq=[]
        schannels=inChannels
        sstrides=strides
        
        for su in range(subunits):
            seq.append(Convolution2D(schannels,outChannels,sstrides,kernelsize,normalizeFunc))
            schannels=outChannels # after first loop set the channels and strides to what they should be for subsequent units
            sstrides=1
            
        self.conv=nn.Sequential(*seq) # apply this sequence of operations to the input
        
        # apply this convolution to the input to change the number of output channels and output size to match that coming from self.conv
        self.residual=nn.Conv2d(inChannels,outChannels,kernel_size=kernelsize,stride=strides,padding=padding)
        
    def forward(self,x):
        res=self.residual(x) # create the additive residual from x
        cx=self.conv(x) # apply x to sequence of operations
        
        return cx+res # add the residual to the output
    

class UpsampleConcat2D(nn.Module):
    def __init__(self,inChannels,outChannels,strides=1,kernelsize=3):
        super(UpsampleConcat2D,self).__init__()
        padding=strides-1
        self.convt=nn.ConvTranspose2d(inChannels,outChannels,kernelsize,strides,1,padding)
      
    def forward(self,x,y):
        x=self.convt(x)
        #print(x.shape,y.shape)
        return torch.cat([x,y],1)


class ResidualClassifier(nn.Module):
#    def __init__(self,inChannels,classes,channels,strides,kernelsize=3,numSubunits=2,normalizeFunc=None):
#        assert len(channels)==len(strides)
#        self.inChannels=inChannels
#        self.classes=classes
#        self.channels=channels
#        self.strides=strides
#        self.kernelsize=kernelsize
#        self.numSubunits=numSubunits
#        self.normalizeFunc=normalizeFunc
#        
#        modules=[]
#        echannel=inChannels
#        
#        for i,(c,s) in enumerate(zip(channels,strides)):
#            modules.append(('layer_%i'%i,ResidualUnit2D(echannel,c,s,self.kernelsize,self.numSubunits,self.normalizeFunc)))
#            echannel=c
#            
#        modules.append(('end',ResidualUnit2D(echannel,classes,1,self.kernelsize,self.numSubunits,self.normalizeFunc)))
#        
#        self.classifier=nn.Sequential(collections.OrderedDict(modules))
#        
#    def forward(self,x):
#        out=self.classifier(x)
#        #out=out.view(x.shape[0],-1)
#        #assert out.shape==(x.shape[0],self.classes),'%s != %s'%(out.shape,(x.shape[0],self.classes))
#        return (out,)
    def __init__(self,inShape,classes,channels,strides,kernelsize=3,numSubunits=2,normalizeFunc=None):
        super(ResidualClassifier,self).__init__()
        assert len(channels)==len(strides)
        self.inHeight,self.inWidth,self.inChannels=inShape
        self.channels=channels
        self.strides=strides
        self.classes=classes
        self.kernelsize=kernelsize
        self.numSubunits=numSubunits
        self.normalizeFunc=normalizeFunc
        
        modules=[]
        self.linear=None
        echannel=self.inChannels
        
        self.finalSize=np.asarray([self.inHeight,self.inWidth],np.int)
        
        # encode stage
        for i,(c,s) in enumerate(zip(self.channels,self.strides)):
            modules.append(('layer_%i'%i,ResidualUnit2D(echannel,c,s,self.kernelsize,self.numSubunits,self.normalizeFunc)))
            
            echannel=c # use the output channel number as the input for the next loop
            self.finalSize=self.finalSize//s

        self.linear=nn.Linear(int(np.product(self.finalSize))*echannel,self.classes)
        
        self.classifier=nn.Sequential(collections.OrderedDict(modules))
        
    def forward(self,x):
        b=x.size(0)
        x=self.classifier(x)
        x=x.view(b,-1)
        x=self.linear(x)
        return (x,)
        

class AutoEncoder2D(nn.Module):
    def __init__(self,inChannels,outChannels,channels,strides,kernelsize=3,numSubunits=2,normalizeFunc=None):
        super(AutoEncoder2D,self).__init__()
        assert len(channels)==len(strides)
        self.inChannels=inChannels
        self.outChannels=outChannels
        self.channels=channels
        self.strides=strides
        self.kernelsize=kernelsize
        self.numSubunits=numSubunits
        self.normalizeFunc=normalizeFunc
        
        modules=[]
        echannel=inChannels
        
        # encoding stage
        for i,(c,s) in enumerate(zip(channels,strides)):
            modules.append(('encode_%i'%i,ResidualUnit2D(echannel,c,s,self.kernelsize,self.numSubunits,self.normalizeFunc)))
            echannel=c
            
        # decoding stage
        for i,(c,s) in enumerate(zip(list(channels[-2::-1])+[outChannels],strides[::-1])):
            modules+=[
                ('up_%i'%i,nn.ConvTranspose2d(echannel,echannel,self.kernelsize,s,1,s-1)),
                ('decode_%i'%i,ResidualUnit2D(echannel,c,1,self.kernelsize,self.numSubunits,self.normalizeFunc))
            ]
            echannel=c

        self.conv=nn.Sequential(collections.OrderedDict(modules))
        
    def forward(self,x):
        return (self.conv(x),)
    

class Unet2D(nn.Module):
    def __init__(self,inChannels,numClasses,channels,strides,kernelsize=3,numSubunits=2,normalizeFunc=None):
        super(Unet2D,self).__init__()
        assert len(channels)==len(strides)
        self.inChannels=inChannels
        self.numClasses=numClasses
        self.channels=channels
        self.strides=strides
        self.kernelsize=kernelsize
        self.numSubunits=numSubunits
        self.normalizeFunc=normalizeFunc
        
        dchannels=[self.numClasses]+list(self.channels[:-1])

        self.encodes=[] # list of encode stages, this is build up in reverse order so that the decode stage works in reverse
        self.decodes=[]
        echannel=inChannels
        
        # encode stage
        for c,s,dc in zip(self.channels,self.strides,dchannels):
            x=ResidualUnit2D(echannel,c,s,self.kernelsize,self.numSubunits,self.normalizeFunc)
            
            setattr(self,'encode_%i'%(len(self.encodes)),x)
            self.encodes.insert(0,(x,dc,s,echannel))
            echannel=c # use the output channel number as the input for the next loop
            
        # decode stage
        for ex,c,s,ec in self.encodes:
            up=UpsampleConcat2D(ex.outChannels,ex.outChannels,s,self.kernelsize)
            x=ResidualUnit2D(ex.outChannels+ec,c,1,self.kernelsize,1,self.normalizeFunc)
            
            setattr(self,'up_%i'%(len(self.decodes)),up)
            setattr(self,'decode_%i'%(len(self.decodes)),x)
            
            self.decodes.append((up,x))
        
    def forward(self,x):
        elist=[] # list of encode stages, this is build up in reverse order so that the decode stage works in reverse

        # encode stage
        for ex,_,_,_ in reversed(self.encodes):
            i=len(elist)
            addx=x
            x=ex(x)
            elist.insert(0,(addx,)+self.decodes[-i-1])

        # decode stage
        for addx,up,ex in elist:
            x=up(x,addx)
            x=ex(x)
            
        # generate prediction outputs, x has shape BCHW
        if self.numClasses==1:
            preds=(x[:,0]>=0.5).type(torch.IntTensor)
        else:
            preds=x.max(1)[1]

        return x, preds