# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

from __future__ import print_function,division
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


def oneHot(labels,numClasses):
    '''
    For a tensor `labels' of dimensions BC[D][H]W, return a tensor of dimensions BC[D][H]WN for `numClasses' N number of 
    classes. For every value v = labels[b,c,h,w], the value in the result at [b,c,h,w,v] will be 1 and all others 0. 
    Note that this will include the background label, thus a binary mask should be treated as having 2 classes.
    '''
    onehotshape=tuple(labels.shape)+(numClasses,)
    labels=labels%numClasses
    y = torch.eye(numClasses,device=labels.device)
    onehot=y[labels.view(-1).long()]
    
    return onehot.reshape(*onehotshape) 


def samePadding(kernelSize):
    '''
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.
    '''
    kernelSize=np.atleast_1d(kernelSize)
    padding=(kernelSize-1)//2
    
    return tuple(padding) if padding.shape[0]>1 else padding[0]
    

def calculateOutShape(inShape,kernelSize,stride,padding):
    '''
    Calculate the output tensor shape when applying a convolution to a tensor of shape `inShape' with kernel size 
    'kernelSize', stride value `stride', and input padding value `padding'. All arguments can be scalars or multiple
    values, return value is a scalar if 
    '''
    inShape=np.atleast_1d(inShape)
    outShape=((inShape-kernelSize+padding+padding)//stride) +1
    
    return tuple(outShape) if outShape.shape[0]>1 else outShape[0]


def normalInit(m,std=0.02,normalFunc=torch.nn.init.normal_):
    '''
    Initialize the weight and bias tensors of `m' and its submodules to values from a normal distribution with a stddev
    of `std'. Weight tensors of convolution and linear modules are initialized with a mean of 0, batch norm modules with
    a mean of 1. The callable `normalFunc' used to assign values should have the same arguments as its default normal_().
    '''
    cname = m.__class__.__name__
    
    if getattr(m,'weight',None) is not None and (cname.find('Conv') != -1 or cname.find('Linear') != -1):
        normalFunc(m.weight.data, 0.0,std)
        if getattr(m, 'bias',None) is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
                
    elif cname.find('BatchNorm') != -1:
        normalFunc(m.weight.data, 1.0, std)
        torch.nn.init.constant_(m.bias.data, 0)
        
        
def addNormalNoise(m,mean=0,std=1e-5):
    '''Returns `m' added with a normal noise field with given mean and standard deviation values.'''
    noise=torch.zeros_like(m,device=m.device)
    noise.data.normal_(mean,std)
    return m+noise
        

class DiceLoss(_Loss):
    def forward(self, source, target, smooth=1e-5):
        '''
        Multiclass dice loss. Input logits 'source' (BNHW where N is number of classes) is compared with ground truth 
        `target' (B1HW). Axis N of `source' is expected to have logit predictions for each class rather than being the
        image channels, while the same axis of `target' should be 1. If the N channel of `source' is 1 a binary dice loss
        will be calculated. The `smooth' parameter is a value added to the intersection and union components of the 
        inter-over-union calculation to smooth results and prevent divide-by-0, this value should be small.
        '''
        assert target.shape[1]==1,'Target shape is '+str(target.shape)
        
        batchsize = target.size(0)
        
        if source.shape[1]==1: # binary dice loss, use sigmoid activation
            probs=source.float().sigmoid()
            tsum=target
        else:
            # multiclass dice loss, use softmax and convert target to one-hot encoding
            probs=F.softmax(source)
            tsum=oneHot(target,source.shape[1]) # BCHW -> BCHWN
            tsum=tsum[:,0].permute(0,3,1,2).contiguous() # BCHWN -> BNHW
            
            assert tsum.shape==source.shape
        
        tsum = tsum.float().view(batchsize, -1)
        psum = probs.view(batchsize, -1)
        intersection=psum*tsum
        sums=psum+tsum

        score = 2.0 * (intersection.sum(1) + smooth) / (sums.sum(1) + smooth)
        return 1 - score.sum() / batchsize
        

class KLDivLoss(_Loss):
    def __init__(self,reconLoss=torch.nn.BCELoss(reduction='sum'),beta=1.0):
        _Loss.__init__(self)
        self.reconLoss=reconLoss
        self.beta=beta
        
    def forward(self,reconx, x, mu, logvar):
        assert x.min() >= 0. and x.max() <= 1.,'%f -> %f'%(x.min(), x.max() )
        assert reconx.min() >= 0. and reconx.max() <= 1.,'%f -> %f'%(reconx.min(), reconx.max() )
        
        KLD = -0.5 * self.beta * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # KL divergence loss with beta term
        return KLD+self.reconLoss(reconx,x)
        
        
class Convolution2D(nn.Sequential):
    def __init__(self,inChannels,outChannels,strides=1,kernelSize=3,instanceNorm=True,dropout=0,bias=True,convOnly=False):
        super(Convolution2D,self).__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        padding=samePadding(kernelSize)
        normalizeFunc=nn.InstanceNorm2d if instanceNorm else nn.BatchNorm2d
        
        self.add_module('conv',nn.Conv2d(inChannels,outChannels,kernelSize,strides,padding,bias=bias))
        
        if not convOnly:
            self.add_module('norm',normalizeFunc(outChannels))
            if dropout>0:
                self.add_module('dropout',nn.Dropout2d(dropout))
                
            self.add_module('prelu',nn.modules.PReLU())
    
    
class ConvTranspose2D(nn.Sequential):
    def __init__(self,inChannels,outChannels,strides=1,kernelSize=3,instanceNorm=True,dropout=0,bias=True,convOnly=False):
        super(ConvTranspose2D,self).__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        normalizeFunc=nn.InstanceNorm2d if instanceNorm else nn.BatchNorm2d
        
        self.add_module('conv',nn.ConvTranspose2d(inChannels,outChannels,kernelSize,strides,1,strides-1,bias=bias))
        
        if not convOnly:
            self.add_module('norm',normalizeFunc(outChannels))
            if dropout>0:
                self.add_module('dropout',nn.Dropout2d(dropout))
                
            self.add_module('prelu',nn.modules.PReLU())
        

class ResidualUnit2D(nn.Module):
    def __init__(self, inChannels,outChannels,strides=1,kernelSize=3,subunits=2,instanceNorm=True,dropout=0,bias=True,lastConvOnly=False):
        super(ResidualUnit2D,self).__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        self.conv=nn.Sequential()
        
        padding=samePadding(kernelSize)
        schannels=inChannels
        sstrides=strides
        
        for su in range(subunits):
            convOnly=lastConvOnly and su==(subunits-1)
            self.conv.add_module('unit%i'%su,Convolution2D(schannels,outChannels,sstrides,kernelSize,instanceNorm,dropout,bias,convOnly))
            schannels=outChannels # after first loop set the channels and strides to what they should be for subsequent units
            sstrides=1
            
        # apply this convolution to the input to change the number of output channels and output size to match that coming from self.conv
        self.residual=nn.Conv2d(inChannels,outChannels,kernel_size=kernelSize,stride=strides,padding=padding,bias=bias)
        
    def forward(self,x):
        res=self.residual(x) # create the additive residual from x
        cx=self.conv(x) # apply x to sequence of operations
        
        return cx+res # add the residual to the output
    

class ResidualBranchUnit2D(nn.Module):
    def __init__(self, inChannels,outChannels,strides=1,branches=[(3,)],instanceNorm=True,dropout=0):
        super(ResidualBranchUnit2D,self).__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        self.branchSeqs=[]
        
        totalchannels=0
        for i,branch in enumerate(branches):
            seq=[]
            sstrides=strides
            schannels=inChannels
            ochannels=max(1,outChannels//len(branches))
            totalchannels+=ochannels
            
            for kernel in branch:
                seq.append(Convolution2D(schannels,ochannels,sstrides,kernel,instanceNorm,dropout))
                schannels=ochannels # after first conv set the channels and strides to what they should be for subsequent units
                sstrides=1
                
            seq=nn.Sequential(*seq)
            setattr(self,'branch%i'%i,seq)
            self.branchSeqs.append(seq)
            
        # resize branches to have the desired number of output channels
        self.resizeconv=nn.Conv2d(totalchannels,outChannels,kernel_size=1,stride=1)
        
        # apply this convolution to the input to change the number of output channels and output size to match self.resizeconv
        self.residual=nn.Conv2d(inChannels,outChannels,kernel_size=3,stride=strides,padding=samePadding(3))
        
    def forward(self,x):
        res=self.residual(x) # create the additive residual from x
        
        cx=torch.cat([s(x) for s in self.branchSeqs],1)
        cx=self.resizeconv(cx)
        
        return cx+res # add the residual to the output
    

class UpsampleConcat2D(nn.Module):
    def __init__(self,inChannels,outChannels,strides=1,kernelSize=3):
        super(UpsampleConcat2D,self).__init__()
        padding=strides-1
        self.convt=nn.ConvTranspose2d(inChannels,outChannels,kernelSize,strides,1,padding)
      
    def forward(self,x,y):
        x=self.convt(x)
        return torch.cat([x,y],1)


class Classifier(nn.Module):
    def __init__(self,inShape,classes,channels,strides,kernelSize=3,numSubunits=2,instanceNorm=True,dropout=0,bias=True):
        super(Classifier,self).__init__()
        assert len(channels)==len(strides)
        self.inHeight,self.inWidth,self.inChannels=inShape
        self.channels=channels
        self.strides=strides
        self.classes=classes
        self.kernelSize=kernelSize
        self.numSubunits=numSubunits
        self.instanceNorm=instanceNorm
        self.dropout=dropout
        self.bias=bias
        self.classifier=nn.Sequential()
        
        self.linear=None
        echannel=self.inChannels
        
        self.finalSize=np.asarray([self.inHeight,self.inWidth],np.int)
        
        # encode stage
        for i,(c,s) in enumerate(zip(self.channels,self.strides)):
            if numSubunits==0:
                layer=Convolution2D(echannel,c,s,kernelSize,instanceNorm,dropout,bias,i==len(channels)-1)
            else:
                layer=ResidualUnit2D(echannel,c,s,self.kernelSize,self.numSubunits,instanceNorm,dropout)
            
            echannel=c # use the output channel number as the input for the next loop
            self.classifier.add_module('layer_%i'%i,layer)
            self.finalSize=calculateOutShape(self.finalSize,kernelSize,s,samePadding(kernelSize))

        self.linear=nn.Linear(int(np.product(self.finalSize))*echannel,self.classes)
        
    def forward(self,x):
        b=x.size(0)
        x=self.classifier(x)
        x=x.view(b,-1)
        x=self.linear(x)
        return (x,)
    

class Discriminator(Classifier):
    def __init__(self,inShape,channels,strides,kernelSize=3,numSubunits=2,instanceNorm=True,dropout=0,bias=True,lastAct=torch.sigmoid):
        Classifier.__init__(self,inShape,1,channels,strides,kernelSize,numSubunits,instanceNorm,dropout,bias)
        self.lastAct=lastAct
        
    def forward(self,x):
        result=Classifier.forward(self,x)
        
        if self.lastAct is not None:
            result=(self.lastAct(result[0]),)
            
        return result
    

class BranchClassifier(nn.Module):
    def __init__(self,inShape,classes,channels,strides,branches=[(3,)],instanceNorm=True,dropout=0):
        super(BranchClassifier,self).__init__()
        assert len(channels)==len(strides)
        self.inHeight,self.inWidth,self.inChannels=inShape
        self.channels=channels
        self.strides=strides
        self.classes=classes
        self.branches=branches
        self.instanceNorm=instanceNorm
        self.dropout=dropout
        
        modules=[]
        self.linear=None
        echannel=self.inChannels
        
        self.finalSize=np.asarray([self.inHeight,self.inWidth],np.int)
        
        # encode stage
        for i,(c,s) in enumerate(zip(self.channels,self.strides)):
            modules.append(('layer_%i'%i,ResidualBranchUnit2D(echannel,c,s,self.branches,instanceNorm,dropout)))
            
            echannel=c*len(branches) # use the output channel number as the input for the next loop
            self.finalSize=self.finalSize//s

        self.linear=nn.Linear(int(np.product(self.finalSize))*echannel,self.classes)
        
        self.classifier=nn.Sequential(OrderedDict(modules))
        
    def forward(self,x):
        b=x.size(0)
        x=self.classifier(x)
        x=x.view(b,-1)
        x=self.linear(x)
        return (x,)
    
   
class Generator(nn.Module):
    def __init__(self,latentShape,startShape,channels,strides,kernelSize=3,numSubunits=2,instanceNorm=True,dropout=0,bias=True):
        super(Generator,self).__init__()
        assert len(channels)==len(strides)
        self.inHeight,self.inWidth,self.inChannels=tuple(startShape) # HWC
        
        self.latentShape=latentShape
        self.channels=channels
        self.strides=strides
        self.kernelSize=kernelSize
        self.numSubunits=numSubunits
        self.instanceNorm=instanceNorm
        
        echannel=self.inChannels
        self.linear=nn.Linear(np.prod(self.latentShape),int(np.prod(startShape)))
        self.conv=nn.Sequential()
        
        # transform image of shape `startShape' into output shape through transposed convolutions and residual units
        for i,(c,s) in enumerate(zip(channels,strides)):
            isLast=i==len(channels)-1
            
            self.conv.add_module('invconv_%i'%i,ConvTranspose2D(echannel,c,s,kernelSize,instanceNorm,dropout,bias,isLast or numSubunits>0))
            if numSubunits>0:
                self.conv.add_module('decode_%i'%i,ResidualUnit2D(c,c,1,kernelSize,numSubunits,instanceNorm,dropout,bias,isLast))
                
            echannel=c
        
    def forward(self,x):
        b=x.shape[0]
        x=self.linear(x.view(b,-1)).reshape((b,self.inChannels,self.inHeight,self.inWidth))
        return (self.conv(x),)
    

class AutoEncoder(nn.Module):
    def __init__(self,inChannels,outChannels,channels,strides,kernelSize=3,numSubunits=2,instanceNorm=True,dropout=0,numResUnits=0):
        super(AutoEncoder,self).__init__()
        assert len(channels)==len(strides)
        self.inChannels=inChannels
        self.outChannels=outChannels
        self.channels=channels
        self.strides=strides
        self.kernelSize=kernelSize
        self.numSubunits=numSubunits
        self.instanceNorm=instanceNorm
        self.dropout=dropout
        self.numResUnits=numResUnits
        
        self.modules=[]
        echannel=inChannels
        
        # encoding stage
        for i,(c,s) in enumerate(zip(channels,strides)):
            if numSubunits>0:
                unit=ResidualUnit2D(echannel,c,s,kernelSize,self.numSubunits,instanceNorm,dropout)
            else:
                unit=Convolution2D(echannel,c,s,kernelSize,instanceNorm,dropout)
            
            self.modules.append(('encode_%i'%i,unit))
            echannel=c
            
        # intermediate residual units on the bottom path
        for i in range(numResUnits):
            self.modules.append(('res_%i'%i,ResidualUnit2D(echannel,echannel,1,kernelSize,1,instanceNorm,dropout)))
            
        # decoding stage
        for i,(c,s) in enumerate(zip(list(channels[-2::-1])+[outChannels],strides[::-1])):
            isLast= i==(len(strides)-1)
            
            if numSubunits>0:
                self.modules+=[
                    ('up_%i'%i,nn.ConvTranspose2d(echannel,echannel,self.kernelSize,s,1,s-1)),
                    ('decode_%i'%i,ResidualUnit2D(echannel,c,1,self.kernelSize,self.numSubunits,instanceNorm,dropout,isLast))
                ]
            else:
                self.modules.append(('decode_%i'%i,ConvTranspose2D(echannel,c,s,kernelSize,instanceNorm,dropout,True,isLast)))
                
            echannel=c

        self.conv=nn.Sequential(OrderedDict(self.modules))
        
    def forward(self,x):
        return (self.conv(x),)
    
    
class VarAutoEncoder(nn.Module):
    def __init__(self,inShape,latentSize,channels,strides,kernelSize=3,numSubunits=2,instanceNorm=True,dropout=0):
        super(VarAutoEncoder,self).__init__()
        assert len(channels)==len(strides)
        self.inHeight,self.inWidth,self.inChannels=inShape
        self.latentSize=latentSize
        self.channels=channels
        self.strides=strides
        self.kernelSize=kernelSize
        self.numSubunits=numSubunits
        self.instanceNorm=instanceNorm
        
        self.finalSize=np.asarray([self.inHeight,self.inWidth],np.int)
        
        self.encodeModules=OrderedDict()
        self.decodeModules=OrderedDict()
        echannel=self.inChannels
        
        # encoding stage
        for i,(c,s) in enumerate(zip(channels,strides)):
            self.encodeModules['encode_%i'%i]=ResidualUnit2D(echannel,c,s,kernelSize,numSubunits,instanceNorm,dropout)
            #self.encodeModules['encode_%i'%i]=Convolution2D(echannel,c,s,kernelSize,instanceNorm,dropout)
            echannel=c
            self.finalSize=calculateOutShape(self.finalSize,kernelSize,s,samePadding(kernelSize)) #self.finalSize//s
            
        self.encodes=nn.Sequential(self.encodeModules)
        
        linearSize=int(np.product(self.finalSize))*echannel
        self.mu=nn.Linear(linearSize,self.latentSize)
        self.logvar=nn.Linear(linearSize,self.latentSize)
        self.decodeL=nn.Linear(self.latentSize,linearSize)
            
        # decoding stage
        for i,(c,s) in enumerate(zip(list(channels[-2::-1])+[self.inChannels],strides[::-1])):
            self.decodeModules['up_%i'%i]=nn.ConvTranspose2d(echannel,echannel,kernelSize,s,1,s-1)
            self.decodeModules['decode_%i'%i]=ResidualUnit2D(echannel,c,1,kernelSize,numSubunits,instanceNorm,dropout)
            #self.decodeModules['decode_%i'%i]=Convolution2D(echannel,c,1,kernelSize,instanceNorm,dropout)
            echannel=c

        self.decodes=nn.Sequential(self.decodeModules)
        
    def encode(self,x):
        x=self.encodes(x)
        x=x.view(x.shape[0],-1)
        mu=self.mu(x)
        logvar=self.logvar(x)
        return mu,logvar
        
    def decode(self,z):
        x=F.relu(self.decodeL(z))
        x=x.view(x.shape[0],self.channels[-1],self.finalSize[0],self.finalSize[1])
        x=self.decodes(x)
        x=torch.sigmoid(x)
        return x
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
    
class BaseUnet(nn.Module):
    def __init__(self,inChannels,numClasses,channels,strides,upsamplekernelSize=3):
        super(BaseUnet,self).__init__()
        assert len(channels)==len(strides)
        self.inChannels=inChannels
        self.numClasses=numClasses
        self.channels=channels
        self.strides=strides
        self.upsamplekernelSize=upsamplekernelSize
        
        dchannels=[self.numClasses]+list(self.channels[:-1])

        self.encodes=[] # list of encode stages, this is build up in reverse order so that the decode stage works in reverse
        self.decodes=[]
        echannel=inChannels
        
        # encode stage
        for c,s,dc in zip(self.channels,self.strides,dchannels):
            ex=self._getLayer(echannel,c,s,True)
            
            setattr(self,'encode_%i'%(len(self.encodes)),ex)
            self.encodes.insert(0,(ex,dc,s,echannel))
            echannel=c # use the output channel number as the input for the next loop
            
        # decode stage
        for ex,c,s,ec in self.encodes:
            up=self._getUpsampleConcat(echannel,echannel,s,self.upsamplekernelSize)
            x=self._getLayer(echannel+ec,c,1,False)
            echannel=c
            
            setattr(self,'up_%i'%(len(self.decodes)),up)
            setattr(self,'decode_%i'%(len(self.decodes)),x)
            
            self.decodes.append((up,x))
            
    def _getUpsampleConcat(self,inChannels,outChannels,stride,kernelSize):
        return UpsampleConcat2D(inChannels,outChannels,stride,kernelSize)
    
    def _getLayer(self,inChannels,outChannels,strides,isEncode):
        return nn.Conv2d(inChannels,outChannels,3,strides)
        
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
            preds=(x[:,0]>=0).type(torch.IntTensor)
        else:
            preds=x.max(1)[1] # take the index of the max value along dimension 1

        return x, preds


class Unet(BaseUnet):
    def __init__(self,inChannels,numClasses,channels,strides,kernelSize=3,numSubunits=2,instanceNorm=True,dropout=0):
         self.kernelSize=kernelSize
         self.numSubunits=numSubunits
         self.instanceNorm=instanceNorm
         self.dropout=dropout
         super(Unet,self).__init__(inChannels,numClasses,channels,strides,3)

    def _getLayer(self,inChannels,outChannels,strides,isEncode):
        numSubunits=self.numSubunits if isEncode else 1
        return ResidualUnit2D(inChannels,outChannels,strides,self.kernelSize,numSubunits,self.instanceNorm,self.dropout)
    

class BranchUnet(BaseUnet):
    def __init__(self,inChannels,numClasses,channels,strides,branches,instanceNorm=True,dropout=0):
         self.branches=branches
         self.instanceNorm=instanceNorm
         self.dropout=dropout
         super(BranchUnet,self).__init__(inChannels,numClasses,channels,strides,3)
         
    def _getLayer(self,inChannels,outChannels,strides,isEncode):
        return ResidualBranchUnit2D(inChannels,outChannels,strides,self.branches,self.instanceNorm,self.dropout)


        
    
class UnetBlock(nn.Sequential):
    def __init__(self,encode,decode,subblock,isTop=False):
        super(UnetBlock,self).__init__()
        self.isTop=isTop
        self.add_module('encode',encode)
        
        if subblock is not None:
            self.add_module('subblock',subblock)
            
        self.add_module('decode',decode)
        
    def forward(self,x):
        xx=super().forward(x)
        
        if self.isTop:
            return xx
        else:
            return torch.cat([x,xx],1)
        
class BaseUnet1(nn.Module):
    def __init__(self,inChannels,numClasses,channels,strides,upKernelSize=3):
        super(BaseUnet1,self).__init__()
        assert len(channels)==len(strides)
        self.inChannels=inChannels
        self.numClasses=numClasses
        self.channels=channels
        self.strides=strides
        self.upKernelSize=upKernelSize
        self.model=None
        
        def _createBlock(inc,outc,channels,strides,isTop=False):
            if len(channels)==0:
                return None
            
            c=channels[0]
            s=strides[0]
            isBottom=len(channels)==1
            down=self._getDownLayer(inc,c,s)
            up=self._getUpLayer(c*(1 if isBottom else 2),outc,s)
            
            return UnetBlock(down,up,_createBlock(c,c,channels[1:],strides[1:]),isTop)
        
        self.model=_createBlock(inChannels,numClasses,self.channels,self.strides,True)
            
    def forward(self,x):
        x= self.model(x)
        
        # generate prediction outputs, x has shape BCHW
        if self.numClasses==1:
            preds=(x[:,0]>=0).type(torch.IntTensor)
        else:
            preds=x.max(1)[1] # take the index of the max value along dimension 1

        return x, preds
    
    def _getDownLayer(self,inChannels,outChannels,strides):
#        padding=samePadding(3)
#        return nn.Conv2d(inChannels,outChannels,3,strides,padding) 
        return Convolution2D(inChannels,outChannels,strides,3)
    
    def _getUpLayer(self,inChannels,outChannels,strides):
#        padding=strides-1
#        return nn.ConvTranspose2d(inChannels,outChannels,self.upKernelSize,strides,1,padding)
        return ConvTranspose2D(inChannels,outChannels,strides,3)
        
    
if __name__=='__main__':
#    b1=UnetBlock(nn.Conv2d(5,10,3,2,samePadding(3)),nn.ConvTranspose2d(10,5,3,2,1,1),None)
#    b2=UnetBlock(nn.Conv2d(3,5,3,2,samePadding(3)),nn.ConvTranspose2d(5,3,3,2,1,1),b1)
    
#    print(b1(torch.zeros(22,5,16,16)).shape)
#    print(b2(torch.zeros(22,3,16,16)).shape)
    unet=BaseUnet1(1,3,[5,10,15],[2,2,1])
    print(unet)
    print(unet(torch.zeros((2,1,16,16)))[0].shape)