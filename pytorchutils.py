# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file


from __future__ import print_function,division
import os
import glob
import time
import datetime
import threading

import torch
import pytorchnet
import augments
import datasource
import numpy as np


def convertAug(images,out):
    '''Convert `images' and `out' to CH[W] format, assuming `images' is HWC and `out' is H[W].'''
    return images.transpose([2,0,1]), out[np.newaxis,...]


def convertFirst(images,out):
    '''Convert `images' from HWC to CHW format.'''
    return images.transpose([2,0,1]), out


def convertBoth(images,out):
    return images.transpose([2,0,1]), out.transpose([2,0,1])


@augments.augment(prob=1.0)
def convert(*arrs,dims=2):
    """
    Convert arrays to Pytorch format by adding a 1-length first dimension if the ndim of an array is `dims`, or 
    transposing the last dimension to become the first otherwise. This assumes `arrays` lack a channel dimension 
    or have a channel dimension last. The ndim of the members of `arrays` minus the channels dimension must match 
    `dims` to ensure the choice of transform is correct.
    """
    def _convOp(arr):
        if arr.ndim==dims:
            return arr[None]
        else: 
            return np.rollaxis(arr,2)
            
    return _convOp


class SimpleTrainer(object):
    def __init__(self,steps,net,loss,opt=None):
        self.steps = steps
        self.step = 0
        self.net = net
        self._loss = loss
        self.lossval = None
        self.opt = opt or torch.optim.Adam(net.parameters())

    def loss(self, *args, **kwargs):
        self.lossval = self._loss(*args, **kwargs)
        return self.lossval.item()

    def __str__(self):
        lval = self.lossval.item() if self.lossval else float('nan')
        return 'Trainer(Step: %i, Loss: %f)' % (self.step, lval)

    def __iter__(self):
        while self.step < self.steps:
            self.step += 1
            self.opt.zero_grad()
            yield self
            self.lossval.backward()
            self.opt.step()
    

class NetworkManager(object):
    '''
    This manages the training, loading, saving, and evaluation of a input network. It defines the train, evaluate, and
    infer operations in terms of template methods which must be overridden to suit the specific network being managed.
    Information is loaded, saved, and logged to a directory which the manager creates if it doesn't already exist. The
    method descriptions for netForward(), lossForward(), train(), and evaluate() explain the necessary details to 
    implementing a subtype of this class.
    '''
    def __init__(self,net,loss,isCuda=True,opt=None,saveDirPrefix=None,savePrefix='net',loadLastDir=False,**params):
        '''
        Initialize the manager with the given network `net' to be managed, optimizer `opt', and loss function `loss'.
        If `isCuda' is True the network and inputs are converted to cuda tensors. The `saveDirPrefix' is the prefix for
        the new directory to create if it doesn't exist, if it does exist it is expected to be a previously created
        directory with a stored network that is reloaded. The `params' value is a user parameter dict for the network.
        '''
        self.net=net
        self.isCuda=isCuda
        self.device=torch.device('cuda' if isCuda and torch.cuda.is_available else 'cpu')
        self.params=params
        self.opt=opt
        self.loss=loss
        self.step=0
        self.traininputs=None
        self.netoutputs=None
        self.lossoutput=None
        self.isRunning=True
        self.lock=threading.RLock()
        
        self.savedir=None
        self.savePrefix=savePrefix
        self.doLog=True
        
        if isCuda and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if self.net is not None:
            self.net=self.net.to(self.device)
            
        if self.opt is None and self.net is not None:
            lr=params.get('learningRate',1e-3)
            betas=params.get('betas',(0.9, 0.999))
            self.opt=torch.optim.Adam(self.net.parameters(),lr=lr,betas=betas)

        if saveDirPrefix is not None:
            # attempt to choose the most recent saved directory having the given prefix
            if loadLastDir:
                mostRecent=max(glob.glob(saveDirPrefix+'-*'),key=os.path.getctime)
                if os.path.isdir(mostRecent):
                    saveDirPrefix=mostRecent
                
            if os.path.exists(saveDirPrefix): # if the directory exists reload it
                self.savedir=saveDirPrefix
                self.reload()
            else:
                self.setSaveDir(saveDirPrefix)
#                self.savedir='%s-%s'%(saveDirPrefix,datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
#                os.mkdir(self.savedir)
                
    def setSaveDir(self,dirName,addTimestamp=True):
        '''Create a new save directory starting with `dirName', followed by a timestamp suffix if `addTimestamp'.'''
        if addTimestamp:
            dirName='%s-%s'%(dirName,datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
            
        self.savedir=dirName
        os.mkdir(self.savedir)
                
    def updateStep(self,step,steploss):
        '''Called after every train step, with arguments for the step number and loss at that step.'''
        pass
    
    def saveStep(self,step,steploss):
        '''
        Called at every save operation, with arguments for the step number and loss at that step. By default this saves
        the model to a file named for the self.savePrefix value and self.step (ie. ignores `step' and `steploss'). 
        '''
        self.save(os.path.join(self.savedir,'%s_%.6i.pth'%(self.savePrefix,step)))
    
    def evalStep(self,index,steploss,results):
        '''
        Called after every evaluation step, with arguments for the step number and loss at that step. The `results' list
        is the accumulated result from each application of this method. Given self.traininputs and self.netoutputs this
        method is expected to calculate some evaluation result or metric from these and append it to `results'. The
        default implementation simply appends self.netoutputs to `results'.
        '''
        results.append(self.netoutputs)
    
    def netForward(self):
        '''
        Called before every train/evaluate step and is expected to apply self.traininputs to the network's forward pass
        and return the results from that. By default this applies the first element of self.traininputs to self.net and
        returns the result.
        '''
        in0=self.traininputs[0]
        return self.net(in0)
    
    def lossForward(self):
        '''
        Called before every train/evaluate step and is expected to use self.netoutputs to calculate a loss value and 
        return the results from that. By default this applies the last element of self.traininputs and the first element
        of self.netoutputs to self.loss and returns the result.
        '''
        ground=self.traininputs[-1]
        logits=self.netoutputs[0]
        return self.loss(logits,ground) 
    
    def getLogFilename(self):
        return os.path.join(self.savedir,'%s_train.log'%(self.savePrefix,))
                
    def log(self,*items):
        '''Log the given values to getLogFilename().'''
        if self.doLog:
            dt=datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S: ')
            msg=dt+' '.join(map(str,items))
            
            if self.savedir:
                with open(self.getLogFilename(),'a') as o:
                    print(msg,file=o)
    
    def reload(self,prefix=None):
        '''Reload the network state by loading the most recent .pth file in the save directory if there is one.'''
        if self.savedir:
            prefix=self.savePrefix if prefix is None else prefix
            files=glob.glob(os.path.join(self.savedir,prefix+'*.pth'))
            if files:
                self.load(max(files,key=os.path.getctime))
    
    def load(self,path):
        '''Load the network state from the given path.'''
        self.net.load_state_dict(torch.load(path))
    
    def save(self,path):
        '''Save the network state to the given path.'''
        torch.save(self.net.state_dict(),path)
        
    def loadNet(self,path):
        '''Load the network and its state from the given path, value "__net__" in the state dict should be network itself.'''
        state=torch.load(path,map_location='cpu')
        self.net=state.pop('__net__')
        self.net.load_state_dict(state)
        self.net=self.net.to(self.device) # ensure the hardware state of the loaded network matches what's requested
        
    def saveNet(self,path):
        '''Save the network and its state to the given path by adding the network as "__net__" to the state dict.'''
        state=dict(self.net.state_dict())
        state['__net__']=self.net
        torch.save(state,path)
        
    def setRequiresGrad(self,grad=True):
        '''Set requires_grad for every parameter of self.net to `grad'.'''
        for p in self.net.parameters():
            p.requires_grad=grad
        
    def convertArray(self,arr):
        '''Convert the Numpy array `arr' to a PyTorch tensor, converting to Cuda if necessary.'''
        if not isinstance(arr,torch.Tensor):
            arr=torch.from_numpy(arr)
        
        return arr.to(self.device)
    
    def toNumpy(self,arr):
        '''Convert the PyTorch Tensor `arr' to a Numpy array.'''
        return arr.to('cpu').data.numpy()
    
    def trainStep(self,numSubsteps):
        '''
        Implements the basic training sequence for `numSubsteps' number of times. Each sequence is composed of running
        the network forward, running the loss function forward, zeroing optimizer gradients, feeding the loss result
        backward, and stepping the optimizer.
        '''
        for sub in range(numSubsteps):
            self.netoutputs=self.netForward()
            self.lossoutput=self.lossForward()
            
            self.opt.zero_grad()
            self.lossoutput.backward()
            self.opt.step()

    def train(self,inputfunc,steps,substeps=1,savesteps=5):
        '''
        Train the network for `step' number of steps starting at 1, saving `savesteps' number of times at regular 
        intervals. The callable `inputfunc' is expected to take no arguments and return a tuple pf batch Numpy arrays of 
        shape, B, BC, BCHW or BCDHW. A train step is composed of these steps:
            1. `inputfunc' is called, each returned value is converted to a tensor, then tuple of all assigned to self.traininputs
            2. trainStep() is called which is expected to do the following for `substeps' number of times:
              a. self.netForward() is called and results assigned to self.netoutputs
              b. self.lossForward() is called and results assigned to self.lossoutput
              c. The optimizer performs one training step
            3. self.updateStep() is called and the loss value is assigned to self.params['loss']
            4. If the model is saved on the current step, self.save() is used to save then self.saveStep() is called
            
        Throughout the training process self.log() is called regularly to save logging information.
        '''
        self.log('=================================Starting=================================')
        start=time.time()
        
        try:
            assert self.opt is not None
            assert self.loss is not None
            
            self.log('Params:',self.params)
            self.log('Savedir:',self.savedir)
            self.isRunning=True
            
            if self.net is not None:
                self.net.train()
            
            for s in range(1,steps+1):
                self.log('Timestep',s,'/',steps)
                self.step+=1
                
                with self.lock:
                    self.traininputs=[self.convertArray(arr) for arr in inputfunc()] 
                    self.trainStep(substeps)
                
                    lossval=self.lossoutput.item()
                    self.log('Loss:',lossval)
                    self.updateStep(s,lossval)
                    self.params['loss']=lossval
                
                    if self.savedir and savesteps>0 and (not self.isRunning or s==steps or (s%(steps//savesteps))==0):
                        self.saveStep(self.step,lossval)
                    
                if not self.isRunning:
                    break
                    
        except Exception as e:
            self.log(e)
            raise
        finally:
            self.log('Total time (s): %s'%(time.time()-start))
            self.log('Params:',self.params)
            self.log('===================================Done===================================')

    def evaluate(self,inputs,batchSize=2):
        '''
        Evaluate the network by applying it to batches of size `batchSize' from the input arrays given in `inputs'. The
        evaluation process differs from training in that the optimizer is not used and inputs are given together as one
        large chunk rather than from than input source callable. The results are a list of loss values one for each batch
        and a list of outputs one for each batch. The process is:
            1. Convert each batch slice of arrays in `inputs' to Variables and store all in self.traininputs
            2. self.netForward() is called and results assigned to self.netoutputs
            3. self.lossForward() is called and results assigned to self.lossoutput
            4. Call self.evalStep()
            5. Clear the stored variables to free the graph
        '''
        self.log('================================Evaluating================================')
        start=time.time()
        
        try:
            inputlen=inputs[0].shape[0]
            losses=[]
            results=[]
            
            if self.net is not None:
                self.net.eval()
            
            with torch.no_grad():
                for i in range(0,inputlen,batchSize):
                    with self.lock:
                        self.traininputs=[self.convertArray(arr[i:i+batchSize]) for arr in inputs]
                        self.netoutputs=self.netForward()
                        self.lossoutput=self.lossForward()
                        losses.append(self.lossoutput.item())
    
                        self.evalStep(i,losses[-1],results)
                        # clear stored variables to free graph
                        self.traininputs=None
                        self.netoutputs=None
                        self.lossoutput=None
                
        except Exception as e:
            self.log(e)
            raise
        finally:
            if self.net is not None:
                self.net.train()
                
            self.log('Total time (s): %s'%(time.time()-start))
            self.log('Losses:',losses)
            self.log('===================================Done===================================')
            
        return losses,results
    
    def infer(self,inputs,batchSize=2):
        '''
        Infer results by applying it to batches of size `batchSize' from the input arrays given in `inputs'. This only 
        uses the forward pass of the network to compute output and does not compute loss or use the optimizer:
            1. Convert each batch slice of arrays in `inputs' to tensors and store all in self.traininputs
            2. self.netForward() is called and results assigned to self.netoutputs
            3. If self.netoutputs is a list or tuple, each tensor it stores is converted to Numpy and appended to the 
            results list, if a single tensor this is converted then appended
            4. Clear the stored variables to free the graph
            
        The result is a list of converted outputs, one for each batch. 
        '''
        assert all(i.shape[0]==inputs[0].shape[0] for i in inputs)
        inputlen=inputs[0].shape[0]
        results=[]
        
        try:
            if self.net is not None:
                self.net.eval()
                
            with torch.no_grad():
                for i in range(0,inputlen,batchSize):
                    with self.lock:
                        self.traininputs=[self.convertArray(arr[i:i+batchSize]) for arr in inputs]
                        self.netoutputs=self.netForward()
    
                        if isinstance(self.netoutputs,(tuple,list)):
                            results.append(tuple(map(self.toNumpy, self.netoutputs)))
                        else:
                            results.append(self.toNumpy(self.netoutputs))
    
                        self.traininputs=None
                        self.netoutputs=None

            return results
        finally:
            if self.net is not None:
                self.net.train()
    
    
class SegmentMgr(NetworkManager):
    '''
    Basic manager subtype for segmentation, specifying Adam as the optimizer with params['learningRate'] used as
    the learn rate, and a loss function defined as DiceLoss. This expects the first value in self.traininputs to
    be the images and the last to be the masks, and the first value in self.netoutputs to be the logits.
    '''
    def __init__(self,net,isCuda=True,opt=None,saveDirPrefix=None,savePrefix='net',loss=None,**params):
        loss=loss if loss is not None else pytorchnet.DiceLoss()
        super().__init__(net,loss,isCuda,opt,saveDirPrefix,savePrefix,**params)   
    
 
class AutoEncoderMgr(NetworkManager):
    '''
    Basic manager subtype for autoencoders, specifying Adam as the optimizer with params['learningRate'] used as
    the learn rate, and a loss function defined as MSELoss. This expects the first value in self.traininputs to be
    the input images and the last to be the output images, and the first value in self.netoutputs to be the logits.
    '''
    def __init__(self,net,isCuda=True,opt=None,saveDirPrefix=None,savePrefix='net',loss=None,**params):
        loss=loss if loss is not None else torch.nn.MSELoss()
        super().__init__(net,loss,isCuda,opt,saveDirPrefix,savePrefix,**params)
    
    
class VarAutoEncoderMgr(NetworkManager):
    def __init__(self,net,isCuda=True,opt=None,saveDirPrefix=None,savePrefix='net',loss=None,**params):
        loss=loss if loss is not None else pytorchnet.KLDivLoss()
        super().__init__(net,loss,isCuda,opt,saveDirPrefix,savePrefix,**params)
    
    def lossForward(self):
        outs=self.traininputs[-1]
        recon,mu, logvar, _=self.netoutputs
        return self.loss(recon,outs,mu,logvar)
    
    
class ImageClassifierMgr(NetworkManager):
    '''
    Basic manager subtype for classifier networks, specifying Adam as the optimizer with params['learningRate'] used as
    the learn rate, and a loss function defined as CrossEntropyLoss. This expects the first value in self.traininputs to 
    be the input images and the last to be the 1D category labels vector, and the first value in self.netoutputs to be 
    the one-hot vector of predictions, ie. of dimensions BC or (batch,# of categories).
    '''
    def __init__(self,net,isCuda=True,opt=None,saveDirPrefix=None,savePrefix='net',loss=None,**params):
        loss=loss if loss is not None else torch.nn.CrossEntropyLoss()
        super().__init__(net,loss,isCuda,opt,saveDirPrefix,savePrefix,**params)
    
    
class DiscriminatorMgr(NetworkManager):
    realLabel=1
    genLabel=0
    
    def __init__(self,net,realDataSrc,isCuda=True,opt=None,saveDirPrefix=None,savePrefix='net',loss=None,stepOptimizer=True, separateBackward=True,**params):
        '''
        Initialize the manager. Arguments:
         - net: discriminator network
         - realDataSrc: DataSource returning real training images as first value in Pytorch BC[D]HW order, second value 
           is B1 array filled with realLabel.
         - saveDirPrefix: prefix for saving to a directory, this may overwrite other networks if used with a generator
           so set self.saveprefix to something other than the default
         - loss: loss function, if this isn't provided the default is BCELoss
        '''
        self.stepOptimizer=stepOptimizer
        self.separateBackward=separateBackward
        self.generatedBuffer=None
        self.realDataSrc=realDataSrc
        self.genDataSrc=datasource.BufferDataSource()
    
        self.realloss=0
        self.genloss=0
        
        loss=loss if loss is not None else torch.nn.BCELoss()
        super().__init__(net,loss,isCuda,opt,saveDirPrefix,savePrefix,**params)
    
    def trainStep(self,numSubsteps):
        realinputs=self.traininputs # already filled by train()
        geninputs=[self.convertArray(arr) for arr in self.geninputfunc()] 
        
        for s in range(numSubsteps):
            if self.stepOptimizer:
                self.opt.zero_grad()
    
            # train with real images
            self.traininputs=realinputs 
            self.netoutputs=self.netForward()
            self.realloss=self.lossForward()
            
            if self.separateBackward:
                self.realloss.backward()
    
            # train with generated images
            self.traininputs=geninputs
            self.netoutputs=self.netForward()
            self.genloss=self.lossForward()
            
            if self.separateBackward:
                self.genloss.backward()
    
            self.lossoutput=self.realloss+self.genloss
    
            if not self.separateBackward:
                self.lossoutput.backward()
    
            if self.stepOptimizer:
                self.opt.step()
    
    def train(self,realinputfunc,geninputfunc,steps,substeps=1,savesteps=5):
        self.geninputfunc=geninputfunc
        NetworkManager.train(self,realinputfunc,steps,substeps,savesteps)
        self.geninputfunc=None
    
    def trainDiscriminator(self,batchSize,steps,substeps=1,savesteps=5,numThreads=None,clearBuffer=True):
        if self.genDataSrc.bufferSize()>0:
            with self.realDataSrc.threadBatchGen(batchSize,numThreads=numThreads) as realinputfunc:
                with self.genDataSrc.threadBatchGen(batchSize,numThreads=numThreads) as geninputfunc:
                    self.train(realinputfunc,geninputfunc,steps,substeps,savesteps)
                
        if clearBuffer:
            self.genDataSrc.clearBuffer()
    
    def appendGeneratedOutput(self,output):
        '''
        Add images in BCHW order to the buffer of generated images to use for training the discriminator. These are 
        assumed to be generated by the network being discriminated. Data is copied to `output' can be kept by caller.
        '''
        self.genDataSrc.appendBuffer(output,np.full((output.shape[0],1),self.genLabel,np.float32))
                
    def __call__(self,testinput):
        '''Calculate the discriminator loss on images `testinput'.'''
        output=self.net(testinput)[0]
        cats=torch.full((output.shape[0],1), self.realLabel)
        if self.isCuda:
            cats=cats.cuda()
            
        return self.loss(output,cats)
    

class GeneratorMgr(NetworkManager):
    def __init__(self,net,disc,isCuda=True,opt=None,saveDirPrefix=None,savePrefix='net',loss=None,**params):
        self.disc=disc
        if loss is None:
            loss=disc
            
        super().__init__(net,loss,isCuda,opt,saveDirPrefix,savePrefix,**params)
    
    def lossForward(self):
        preds=self.netoutputs[0]
        self.disc.appendGeneratedOutput(self.toNumpy(preds))
        
        discloss=self.loss(preds)
        return discloss
    

class CycleGANMgr(NetworkManager):
    def __init__(self,net,discA,discB,srcA,srcB,discparams, lambdaA=1.0, lambdaB=1.0, lambdaIdent=0.0,
                 loss=torch.nn.MSELoss(), lossIdent=torch.nn.MSELoss(), saveDirPrefix=None,loadLastDir=False,**params):
        
        self.srcA=srcA
        self.srcB=srcB
        self.discparams=discparams
        self.lossVals=[]
        
        self.lambdaIdent=lambdaIdent
        self.lambdaA=lambdaA
        self.lambdaB=lambdaB
        self.lossIdent=lossIdent
        self.discA=discA
        self.discB=discB
        
        NetworkManager.__init__(self,net,loss,saveDirPrefix=saveDirPrefix,loadLastDir=loadLastDir,**params)

        if self.discA.savedir is None:
            self.discA.savedir=self.savedir
            if self.discA.savePrefix==self.savePrefix:
                self.discA.savePrefix='discA'
                
        if self.discB.savedir is None:
            self.discB.savedir=self.savedir
            if self.discB.savePrefix==self.savePrefix:
                self.discB.savePrefix='discB'
        
    def netForward(self):
        imA,_,imB,_=self.traininputs
        return self.net(imA,imB)
    
    def saveStep(self,step,steploss):
        self.discA.saveStep(step,steploss)
        self.discB.saveStep(step,steploss)
        super().saveStep(step,steploss)
    
    def trainStep(self,numSubsteps):
        super().trainStep(numSubsteps)
        
        outA,outB,reconA,reconB=self.netoutputs
        
        self.discA.setRequiresGrad(True)
        self.discB.setRequiresGrad(True)
        
        self.discA.appendGeneratedOutput(self.toNumpy(outA))
        self.discB.appendGeneratedOutput(self.toNumpy(outB))
        
        discBatchSize=self.discparams.get('batchSize',self.traininputs[0].shape[0])
        discTrainSteps=self.discparams.get('trainSteps',10)
        discSubSteps=self.discparams.get('substeps',5)

        self.discA.trainDiscriminator(discBatchSize,discTrainSteps,discSubSteps,0)
        self.discB.trainDiscriminator(discBatchSize,discTrainSteps,discSubSteps,0)

    def lossForward(self):
        imA,_,imB,_=self.traininputs
        outA,outB,reconA,reconB=self.netoutputs
        
        self.discA.setRequiresGrad(False)
        self.discB.setRequiresGrad(False)
        
        self.lossVals=[
            self.loss(reconA,imA)*self.lambdaA,
            self.loss(reconB,imB)*self.lambdaB,
            self.discA(outA),
            self.discB(outB)
        ]
        
        # identity mapping loss
        if self.lambdaIdent>0:
            identB=self.net.b2aForward(imA)
            identA=self.net.a2bForward(imB)
            self.lossVals+=[
                self.lossIdent(identA,imB)*self.lambdaB*self.lambdaIdent,
                self.lossIdent(identB,imA)*self.lambdaA*self.lambdaIdent
            ]
        
        return sum(self.lossVals)
    