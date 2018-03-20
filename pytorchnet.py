
from __future__ import print_function,division
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def reduceWith(func,inp,axes=None):
    axes=axes or range(len(inp.shape))
    for a in sorted(axes,reverse=True):
        inp=func(inp,a)
        
    return inp


class BinaryDiceLoss(_Loss):
    def forward(self, source, target, smooth=1e-5):
        axis = list(range(2, len(source.shape))) # for BCWH sum over WH
        
        source=source.double()
        target=target.double()
        
        probs=source.sigmoid()
        psum=reduceWith(torch.sum,probs,axis)
        tsum=reduceWith(torch.sum,target,axis)
        
        inter=reduceWith(torch.sum,target*probs,axis)
        sums=psum+tsum
        
        dice=reduceWith(torch.mean,(2.0 * inter + smooth) / (sums + smooth))
        
        return 1.0-dice
    

class ResidualUnit2D(nn.Module):
    def __init__(self, inChannels,outChannels,strides=1,kernelsize=3,subunits=1):
        super(ResidualUnit2D,self).__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        
        padding=(kernelsize-1)//2 
        seq=[]
        schannels=inChannels
        sstrides=strides
        
        for su in range(subunits):
            seq+=[
                nn.BatchNorm2d(schannels), 
                nn.modules.PReLU(),
                nn.Conv2d(schannels,outChannels,kernel_size=kernelsize,stride=sstrides,padding=padding)
            ]
            schannels=outChannels # after first loop set the channels and strides to what they should be for subsequent units
            sstrides=1
            
        self.conv=nn.Sequential(*seq) # apply this sequence of operations to the input
        
        # apply this convolution to the input to change the number of output channels to match that coming from self.conv
        self.residual=nn.Conv2d(inChannels,outChannels,kernel_size=kernelsize,stride=1,padding=padding)
        
        # if the input was strided down then self.res should apply maxpool to the input first before changing output channels
        if strides!=1: 
            self.residual=nn.Sequential(nn.MaxPool2d(kernelsize,strides,padding),self.residual)
        
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


class Unet2D(nn.Module):
    def __init__(self,inChannels,numClasses,channels,strides,kernelsize=3,numSubunits=2):
        super(Unet2D,self).__init__()
        assert len(channels)==len(strides)
        self.inChannels=inChannels
        self.numClasses=numClasses
        self.channels=channels
        self.strides=strides
        self.kernelsize=kernelsize
        self.numSubunits=numSubunits
        
        dchannels=[self.numClasses]+list(self.channels[:-1])

        self.encodes=[] # list of encode stages, this is build up in reverse order so that the decode stage works in reverse
        self.decodes=[]
        echannel=inChannels
        
        # encode stage
        for c,s,dc in zip(self.channels,self.strides,dchannels):
            x=ResidualUnit2D(echannel,c,s,self.kernelsize,self.numSubunits)
            
            setattr(self,'encode_%i'%(len(self.encodes)),x)
            self.encodes.insert(0,(x,dc,s,echannel))
            echannel=c
            
        # decode stage
        for ex,c,s,ec in self.encodes:
            up=UpsampleConcat2D(ex.outChannels,ex.outChannels,s,self.kernelsize)
            x=ResidualUnit2D(ex.outChannels+ec,c,1,self.kernelsize,1)
            
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
            
        # generate prediction outputs, x has shape (B,H,W,numClasses)
        if self.numClasses==1:
            preds=(x[:,0]>=0.5).type(torch.IntTensor)
        else:
            preds=x.max(1)[1]

        return x, preds
    
