# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

from __future__ import print_function,division
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


def samePadding(kernelSize,dilation=1):
    '''
    Return the padding value needed to ensure a convolution using the given kernel size produces an output of the same
    shape as the input for a stride of 1, otherwise ensure a shape of the input divided by the stride rounded down.
    '''
    kernelSize=np.atleast_1d(kernelSize)
    padding=((kernelSize-1)//2)+(dilation-1)
    
    return tuple(padding) if padding.shape[0]>1 else padding[0]
    

def calculateOutShape(inShape,kernelSize,stride,padding):
    '''
    Calculate the output tensor shape when applying a convolution to a tensor of shape `inShape' with kernel size 
    'kernelSize', stride value `stride', and input padding value `padding'. All arguments can be scalars or multiple
    values, return value is a scalar if all inputs are scalars.
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


def predictSegmentation(logits):
    '''
    Given the logits from a network, computing the segmentation by thresholding all values above 0 if `logits' has one
    channel, or computing the argmax along the channel axis otherwise.
    '''
    # generate prediction outputs, logits has shape BCHW[D]
    if logits.shape[1]==1:
        return (logits[:,0]>=0).type(torch.IntTensor) # for binary segmentation threshold on channel 0
    else:
        return logits.max(1)[1] # take the index of the max value along dimension 1
        

class DiceLoss(_Loss):
    '''
    Multiclass dice loss. Input logits 'source' (BNHW where N is number of classes) is compared with ground truth 
    `target' (B1HW). Axis N of `source' is expected to have logit predictions for each class rather than being the image 
    channels, while the same axis of `target' should be 1. If the N channel of `source' is 1 binary dice loss will be 
    calculated. The `smooth' parameter is a value added to the intersection and union components of the inter-over-union 
    calculation to smooth results and prevent divide-by-0, this value should be small. The `includeBackground' class
    attribute can be set to False for an instance of DiceLoss to exclude the first category (channel index 0) which is
    by convention assumed to be the background. If the non-background segmentations are small compared to the total image
    size they can get overwhelmed by the signal from the background so excluding it in such cases helps convergence.
    '''
    
    includeBackground = True  # set to False to exclude the background category (channel index 0) from the loss calculation

    def forward(self, source, target, smooth=1e-5):
        assert target.shape[1] == 1, 'Target should have only a single channel, shape is ' + str(target.shape)

        if source.shape[1] == 1:  # binary dice loss, use sigmoid activation
            psum = source.float().sigmoid()
            tsum = target
        else:
            # multiclass dice loss, use softmax in the first dimension and convert target to one-hot encoding
            psum = F.softmax(source, 1)
            tsum = oneHot(target, source.shape[1])  # BCHW -> BCHWN
            tsum = tsum[:, 0].permute(0, 3, 1, 2).contiguous()  # BCHWN -> BNHW

            assert tsum.shape == source.shape, 'One-hot encoding of target has differing shape from source'

            # exclude background category so that it doesn't overwhelm the other segmentations if they are small
            if not self.includeBackground:
                tsum = tsum[:, 1:]
                psum = psum[:, 1:]
                source = source[:, 1:]

        batchsize = target.size(0)
        tsum = tsum.float().view(batchsize, -1)
        psum = psum.view(batchsize, -1)
        
        intersection = psum * tsum
        sums = psum + tsum

        score = 2.0 * (intersection.sum(1) + smooth) / (sums.sum(1) + smooth)
        return 1 - score.sum() / batchsize
        

class KLDivLoss(_Loss):
    '''
    Computes a loss combining KL divergence and a reconstruction loss. The default reconstruction loss is BCE which is
    suited for images, substituting this for L1/L2 loss for regression appears to work better for smaller data sets. The
    balance between KLD and the recon loss is important, so the KLD weight value `beta' should be adjusted to prevent 
    KLD dominating the output.
    '''
    def __init__(self,reconLoss=torch.nn.BCELoss(reduction='sum'),beta=1.0):
        '''Initialize the loss function with the given reconstruction loss function `reconloss' and KLD weight `beta'.'''
        super().__init__()
        self.reconLoss=reconLoss
        self.beta=beta
        
    def forward(self,reconx,x,mu,logvar):
#         assert 0.0<=x.min()<x.max()<=1.0,'%f -> %f'%(x.min(), x.max())
#         assert 0.0<=reconx.min()<reconx.max()<=1.0,'%f -> %f'%(reconx.min(), reconx.max())
        
        KLD = -0.5*self.beta*torch.sum(1+logvar-mu.pow(2)-logvar.exp()) # KL divergence loss with beta term
        return KLD+self.reconLoss(reconx,x)
        
    
class ThresholdMask(torch.nn.Module):
    '''Binary threshold layer which converts all values above the given threshold to 1 and all below to 0.'''
    def __init__(self,thresholdValue=0):
        super().__init__()
        self.threshold=nn.Threshold(thresholdValue,0)
        self.eps=1e-10
        
    def forward(self,logits):
        t=self.threshold(logits)
        return (t/(t+self.eps))/(1-self.eps)
    
        
class DNN(torch.nn.Module):
    '''Plain dense neural network of linear layers using dropout and PReLU activation.'''
    def __init__(self,inChannels,outChannels,hiddenChannels,dropout=0,bias=True):
        '''
        Defines a network accept input with `inChannels' channels, output of `outChannels' channels, and hidden layers 
        with channels given in `hiddenChannels'. If `bias' is True then linear units have a bias term.
        '''
        super().__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        self.dropout=dropout
        self.hiddenChannels=list(hiddenChannels)
        self.hiddens=torch.nn.Sequential()
        
        prevChannels=self.inChannels
        for i,c in enumerate(hiddenChannels):
            self.hiddens.add_module('hidden_%i'%i,self._getLayer(prevChannels,c,bias))
            prevChannels=c
            
        self.output=torch.nn.Linear(prevChannels,outChannels,bias)
        
    def _getLayer(self,inChannels,outChannels,bias):
        return torch.nn.Sequential(
            torch.nn.Linear(inChannels,outChannels,bias),
            torch.nn.Dropout(self.dropout),
            torch.nn.PReLU()
        )
    
    def forward(self,x):
        b=x.shape[0]
        x=x.view(b,-1)
        x=self.hiddens(x)
        x=self.output(x)
        return x
    

class DenseVAE(torch.nn.Module):
    # like https://github.com/pytorch/examples/blob/master/vae/main.py but configurable through the constructor
    
    def __init__(self,inChannels,outChannels,latentSize,encodeChannels,decodeChannels,dropout=0,bias=True):
        super().__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        self.latentSize=latentSize
        self.dropout=dropout
        
        self.encode=nn.Sequential()
        self.decode=nn.Sequential()
        
        prevChannels=self.inChannels
        for i,c in enumerate(encodeChannels):
            self.encode.add_module('encode_%i'%i,self._getLayer(prevChannels,c,bias))
            prevChannels=c

        self.mu=nn.Linear(prevChannels,self.latentSize)
        self.logvar=nn.Linear(prevChannels,self.latentSize)
        self.decodeL=nn.Linear(self.latentSize,prevChannels)
        
        for i,c in enumerate(decodeChannels):
            self.decode.add_module('decode%i'%i,self._getLayer(prevChannels,c,bias))
            prevChannels=c
            
        self.decode.add_module('final',nn.Linear(prevChannels,outChannels,bias))
        
    def _getLayer(self,inChannels,outChannels,bias):
        return torch.nn.Sequential(
            nn.Linear(inChannels,outChannels,bias),
            nn.Dropout(self.dropout),
            nn.PReLU()
        )
        
    def encodeForward(self,x):
        x=self.encode(x)
        x=x.view(x.shape[0],-1)
        mu=self.mu(x)
        logvar=self.logvar(x)
        return mu,logvar
        
    def decodeForward(self,z):
        x=torch.relu(self.decodeL(z))
        x=x.view(x.shape[0],-1)
        x=self.decode(x)
        x=torch.sigmoid(x)
        return x
        
    def reparameterize(self, mu, logvar):
        std=torch.exp(0.5*logvar)
        
        if self.training: # multiply random noise with std only during training
            std=torch.randn_like(std).mul(std)
            
        return std.add_(mu)
        
    def forward(self, x):
        mu, logvar = self.encodeForward(x)
        z = self.reparameterize(mu, logvar)
        return self.decodeForward(z), mu, logvar, z
    
        
class Convolution2D(nn.Sequential):
    def __init__(self,inChannels,outChannels,strides=1,kernelSize=3,instanceNorm=True,
                 dropout=0,dilation=1,bias=True,convOnly=False,isTransposed=False):
        super(Convolution2D,self).__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        self.isTransposed=isTransposed
        
        padding=samePadding(kernelSize,dilation)
        normalizeFunc=nn.InstanceNorm2d if instanceNorm else nn.BatchNorm2d
        
        if isTransposed:
            conv=nn.ConvTranspose2d(inChannels,outChannels,kernelSize,strides,padding,strides-1,1,bias,dilation)
        else:
            conv=nn.Conv2d(inChannels,outChannels,kernelSize,strides,padding,dilation,bias=bias)
        
        self.add_module('conv',conv)
        
        if not convOnly:
            self.add_module('norm',normalizeFunc(outChannels))
            if dropout>0: # omitting Dropout2d appears faster than relying on it short-circuiting when dropout==0
                self.add_module('dropout',nn.Dropout2d(dropout))
                
            self.add_module('prelu',nn.modules.PReLU())
        

class ResidualUnit2D(nn.Module):
    def __init__(self, inChannels,outChannels,strides=1,kernelSize=3,subunits=2,
                 instanceNorm=True,dropout=0,dilation=1,bias=True,lastConvOnly=False):
        super(ResidualUnit2D,self).__init__()
        self.inChannels=inChannels
        self.outChannels=outChannels
        self.conv=nn.Sequential()
        self.residual=None
        
        padding=samePadding(kernelSize,dilation)
        schannels=inChannels
        sstrides=strides
        subunits=max(1,subunits)
        
        for su in range(subunits):
            convOnly=lastConvOnly and su==(subunits-1)
            unit=Convolution2D(schannels,outChannels,sstrides,kernelSize,instanceNorm,dropout,dilation,bias,convOnly)
            self.conv.add_module('unit%i'%su,unit)
            schannels=outChannels # after first loop set channels and strides to what they should be for subsequent units
            sstrides=1
            
        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides)!=1 or inChannels!=outChannels:
            rkernelSize=kernelSize
            rpadding=padding
            
            if np.prod(strides)==1: # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernelSize=1 
                rpadding=0
                
            self.residual=nn.Conv2d(inChannels,outChannels,rkernelSize,strides,rpadding,bias=bias)
        
    def forward(self,x):
        res=x if self.residual is None else self.residual(x) # create the additive residual from x
        cx=self.conv(x) # apply x to sequence of operations
        return cx+res # add the residual to the output


class Classifier(nn.Module):
    def __init__(self,inShape,classes,channels,strides,kernelSize=3,numResUnits=2,instanceNorm=True,dropout=0,bias=True):
        super(Classifier,self).__init__()
        assert len(channels)==len(strides)
        self.inHeight,self.inWidth,self.inChannels=inShape
        self.channels=channels
        self.strides=strides
        self.classes=classes
        self.kernelSize=kernelSize
        self.numResUnits=numResUnits
        self.instanceNorm=instanceNorm
        self.dropout=dropout
        self.bias=bias
        self.classifier=nn.Sequential()
        
        self.linear=None
        echannel=self.inChannels
        
        self.finalSize=np.asarray([self.inHeight,self.inWidth],np.int)
        
        # encode stage
        for i,(c,s) in enumerate(zip(self.channels,self.strides)):
            layer=self._getLayer(echannel,c,s,i==len(channels)-1)
            echannel=c # use the output channel number as the input for the next loop
            self.classifier.add_module('layer_%i'%i,layer)
            self.finalSize=calculateOutShape(self.finalSize,kernelSize,s,samePadding(kernelSize))

        self.linear=nn.Linear(int(np.product(self.finalSize))*echannel,self.classes)
        
    def _getLayer(self,inChannels,outChannels,strides,isLast):
        if self.numResUnits>0:
            return ResidualUnit2D(inChannels,outChannels,strides,self.kernelSize,
                                  self.numResUnits,self.instanceNorm,self.dropout,1,self.bias,isLast)
        else:
            return Convolution2D(inChannels,outChannels,strides,self.kernelSize,
                                 self.instanceNorm,self.dropout,1,self.bias,isLast)
        
    def forward(self,x):
        b=x.size(0)
        x=self.classifier(x)
        x=x.view(b,-1)
        x=self.linear(x)
        return (x,)
    

class Discriminator(Classifier):
    def __init__(self,inShape,channels,strides,kernelSize=3,numResUnits=2,
                 instanceNorm=True,dropout=0,bias=True,lastAct=torch.sigmoid):
        Classifier.__init__(self,inShape,1,channels,strides,kernelSize,numResUnits,instanceNorm,dropout,bias)
        self.lastAct=lastAct
        
    def forward(self,x):
        result=Classifier.forward(self,x)
        
        if self.lastAct is not None:
            result=(self.lastAct(result[0]),)
            
        return result
    
   
class Generator(nn.Module):
    def __init__(self,latentShape,startShape,channels,strides,kernelSize=3,numSubunits=2,
                 instanceNorm=True,dropout=0,bias=True):
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
            
            conv=Convolution2D(echannel,c,s,kernelSize,instanceNorm,dropout,1,bias,isLast or numSubunits>0,True)
            self.conv.add_module('invconv_%i'%i,conv)
            echannel=c
            
            if numSubunits>0:
                ru=ResidualUnit2D(c,c,1,kernelSize,numSubunits,instanceNorm,dropout,1,bias,isLast)
                self.conv.add_module('decode_%i'%i,ru)
        
    def forward(self,x):
        b=x.shape[0]
        x=self.linear(x.view(b,-1)).reshape((b,self.inChannels,self.inHeight,self.inWidth))
        return (self.conv(x),)
    

class AutoEncoder(nn.Module):
    def __init__(self,inChannels,outChannels,channels,strides,kernelSize=3,upKernelSize=3,
                 numResUnits=0,interChannels=[], interDilations=[], numInterUnits=2,instanceNorm=True, dropout=0):
        super().__init__()
        assert len(channels)==len(strides)
        self.inChannels=inChannels
        self.outChannels=outChannels
        self.channels=list(channels)
        self.strides=list(strides)
        self.kernelSize=kernelSize
        self.upKernelSize=upKernelSize
        self.numResUnits=numResUnits
        self.instanceNorm=instanceNorm
        self.dropout=dropout
        self.numInterUnits=numInterUnits
        self.interChannels=list(interChannels)
        self.interDilations=list(interDilations or [1]*len(interChannels))
        
        self.encodedChannels=inChannels
        self.encode,self.encodedChannels=self._getEncodeModule(self.encodedChannels,channels,strides)
        self.intermediate,self.encodedChannels=self._getIntermediateModule(self.encodedChannels,numInterUnits)
        self.decode,_=self._getDecodeModule(self.encodedChannels,list(channels[-2::-1])+[outChannels],strides[::-1] or [1])
            
    def _getEncodeModule(self,inChannels,channels,strides):
        encode=nn.Sequential()
        layerChannels=inChannels
        
        for i,(c,s) in enumerate(zip(channels,strides)):
            layer=self._getEncodeLayer(layerChannels,c,s,False)
            encode.add_module('encode_%i'%i,layer)
            layerChannels=c
            
        return encode,layerChannels
    
    def _getIntermediateModule(self,inChannels,numInterUnits):
        intermediate=None
        layerChannels=inChannels
            
        if self.interChannels:
            intermediate=nn.Sequential()
            
            for i,(dc,di) in enumerate(zip(self.interChannels,self.interDilations)):
                
                unit=ResidualUnit2D(layerChannels,dc,1,self.kernelSize,
                                self.numInterUnits,self.instanceNorm,self.dropout,di)
                intermediate.add_module('inter_%i'%i,unit)
                layerChannels=dc

        return intermediate,layerChannels
    
    def _getDecodeModule(self,inChannels,channels,strides):
        decode=nn.Sequential()
        layerChannels=inChannels
        
        for i,(c,s) in enumerate(zip(channels,strides)):
            layer=self._getDecodeLayer(layerChannels,c,s,i==(len(strides)-1))
            decode.add_module('decode_%i'%i,layer)
            layerChannels=c
            
        return decode,layerChannels

    def _getEncodeLayer(self,inChannels,outChannels,strides,isLast):
        if self.numResUnits>0:
            return ResidualUnit2D(inChannels,outChannels,strides,self.kernelSize,
                                  self.numResUnits,self.instanceNorm,self.dropout,lastConvOnly=isLast)
        else:
            return Convolution2D(inChannels,outChannels,strides,self.kernelSize,self.instanceNorm,self.dropout,convOnly=isLast)
    
    def _getDecodeLayer(self,inChannels,outChannels,strides,isLast):
        conv=Convolution2D(inChannels,outChannels,strides,self.upKernelSize,
                           self.instanceNorm,self.dropout,convOnly=isLast and self.numResUnits==0,isTransposed=True)
        
        if self.numResUnits>0:
            return nn.Sequential( conv,
                ResidualUnit2D(outChannels,outChannels,1,self.kernelSize,1,self.instanceNorm,
                               self.dropout,lastConvOnly=isLast)
            )
        else:
            return conv
        
    def forward(self,x):
        x=self.encode(x)
        if self.intermediate is not None:
            x=self.intermediate(x)
        x=self.decode(x)
        return (x,)
    
    
class VarAutoEncoder(AutoEncoder):
    def __init__(self,inShape,outChannels,latentSize,channels,strides,kernelSize=3,upKernelSize=3,
                 numResUnits=0, interChannels=[], interDilations=[],numInterUnits=2, instanceNorm=True, dropout=0):
        
        self.inHeight,self.inWidth,inChannels=inShape
        self.latentSize=latentSize
        self.finalSize=np.asarray([self.inHeight,self.inWidth],np.int)
        
        super().__init__(inChannels,outChannels,channels,strides,kernelSize,upKernelSize,numResUnits, 
                         interChannels, interDilations,numInterUnits, instanceNorm, dropout)

        for s in strides:
            self.finalSize=calculateOutShape(self.finalSize,self.kernelSize,s,samePadding(self.kernelSize))

        linearSize=int(np.product(self.finalSize))*self.encodedChannels
        self.mu=nn.Linear(linearSize,self.latentSize)
        self.logvar=nn.Linear(linearSize,self.latentSize)
        self.decodeL=nn.Linear(self.latentSize,linearSize)
        
    def encodeForward(self,x):
        x=self.encode(x)
        if self.intermediate is not None:
            x=self.intermediate(x)
        x=x.view(x.shape[0],-1)
        mu=self.mu(x)
        logvar=self.logvar(x)
        return mu,logvar
        
    def decodeForward(self,z):
        x=F.relu(self.decodeL(z))
        x=x.view(x.shape[0],self.channels[-1],self.finalSize[0],self.finalSize[1])
        x=self.decode(x)
        x=torch.sigmoid(x)
        return x
        
    def reparameterize(self, mu, logvar):
        std=torch.exp(0.5*logvar)
        
        if self.training: # multiply random noise with std only during training
            std=torch.randn_like(std).mul(std)
            
        return std.add_(mu)
        
    def forward(self, x):
        mu, logvar = self.encodeForward(x)
        z = self.reparameterize(mu, logvar)
        return self.decodeForward(z), mu, logvar, z
        
   
class SegnetAE(AutoEncoder):
    def __init__(self,inChannels,numClasses,channels,strides,kernelSize=3,upKernelSize=3,
                 numResUnits=0,interChannels=[], interDilations=[], numInterUnits=2,instanceNorm=True, dropout=0):
        super().__init__(inChannels,numClasses,channels,strides,kernelSize,upKernelSize,
                 numResUnits,interChannels, interDilations, numInterUnits,instanceNorm, dropout)

    def forward(self,x):
        x=super().forward(x)[0]
        return x, predictSegmentation(x)
    
     
class UnetBlock(nn.Module):
    def __init__(self,encode,decode,subblock):
        super().__init__()
        self.encode=encode
        self.decode=decode
        self.subblock=subblock
        
    def forward(self,x):
        enc=self.encode(x)
        sub=self.subblock(enc)
        dec=torch.cat([enc,sub],1)
        return self.decode(dec)
    
  
class Unet(nn.Module):
    def __init__(self,inChannels,numClasses,channels,strides,kernelSize=3,
                 upKernelSize=3,numResUnits=0,instanceNorm=True,dropout=0):
        super().__init__()
        assert len(channels)==(len(strides)+1)
        self.inChannels=inChannels
        self.numClasses=numClasses
        self.channels=channels
        self.strides=strides
        self.kernelSize=kernelSize
        self.upKernelSize=upKernelSize
        self.numResUnits=numResUnits
        self.instanceNorm=instanceNorm
        self.dropout=dropout
        
        def _createBlock(inc,outc,channels,strides,isTop):
            c=channels[0]
            s=strides[0]
            
            if len(channels)>2:
                subblock=_createBlock(c,c,channels[1:],strides[1:],False)
                upc=c*2
            else:
                subblock=self._getBottomLayer(c,channels[1])
                upc=c+channels[1]
                
            down=self._getDownLayer(inc,c,s,isTop)
            up=self._getUpLayer(upc,outc,s,isTop)
            
            return UnetBlock(down,up,subblock)
        
        self.model=_createBlock(inChannels,numClasses,self.channels,self.strides,True)
    
    def _getDownLayer(self,inChannels,outChannels,strides,isTop):
        if self.numResUnits>0:
            return ResidualUnit2D(inChannels,outChannels,strides,self.kernelSize,
                                  self.numResUnits,self.instanceNorm,self.dropout)
        else:
            return Convolution2D(inChannels,outChannels,strides,self.kernelSize,self.instanceNorm,self.dropout)
        
    def _getBottomLayer(self,inChannels,outChannels):
        return self._getDownLayer(inChannels,outChannels,1,False)
    
    def _getUpLayer(self,inChannels,outChannels,strides,isTop):
        conv=Convolution2D(inChannels,outChannels,strides,self.upKernelSize,
                           self.instanceNorm,self.dropout,convOnly=isTop and self.numResUnits==0,isTransposed=True)
        
        if self.numResUnits>0:
            return nn.Sequential( conv,
                ResidualUnit2D(outChannels,outChannels,1,self.kernelSize,1,self.instanceNorm,
                               self.dropout,lastConvOnly=isTop)
            )
        else:
            return conv
            
    def forward(self,x):
        x=self.model(x)
        return x, predictSegmentation(x) 
    
    
if __name__=='__main__':
    t=torch.rand((10,1,256,256))
    
#    b1=UnetBlock(nn.Conv2d(5,10,3,2,samePadding(3)),nn.ConvTranspose2d(10,5,3,2,1,1),None)
#    b2=UnetBlock(nn.Conv2d(3,5,3,2,samePadding(3)),nn.ConvTranspose2d(10,3,3,2,1,1),b1,True)
    
#    print(b1(torch.zeros(22,5,16,16)).shape)
#    print(b2(torch.zeros(22,3,16,16)).shape)
    
#    unet=Unet(1,3,[5,10,15,20],[2,2,2,2])
#    print(unet)
#    print(unet(torch.zeros((2,1,16,16)))[0].shape)
    
#    a=AutoEncoder(1,1,[32, 64, 128],[1,2,2],5,3,0,0)
#    print(t.shape,a(t)[0].shape)
    
#    d=Discriminator((256,256,1),(8,16,32),(2,2,2),3,0,lastAct=None)
#    print(t.shape,d(t)[0].shape)
    
#    t=torch.rand((10,1,32,32))
#    print(t.shape,ConvTranspose2D(1,1,2,2)(t).shape)
#    print(t.shape,ConvTranspose2D(1,1,2,3)(t).shape)
#    print(t.shape,ConvTranspose2D(1,1,2,4)(t).shape)
#    print(t.shape,ConvTranspose2D(1,1,2,5)(t).shape)
    
#    r=ResidualUnit2D(1,2,1)
#    print(r(torch.rand(10,1,32,32)).shape)
    
#    dae=DiAutoEncoder(1,2,[2,4],[2,4])
#    print(dae)
#    print(dae(t)[0].shape)
    