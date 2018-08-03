# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file


from __future__ import print_function,division
import os
import glob
import time
import datetime

import torch
import pytorchnet
import numpy as np


if torch.__version__>='0.4':
    def toItem(t):
        return t.item()
else:
    def toItem(t):
        return t.data[0]
    

def convertAug(images,out):
    '''Convert `images' and `out' to CH[W] format, assuming `images' is HWC and `out' is H[W].'''
    return images.transpose([2,0,1]), out[np.newaxis,...]


def convertFirst(images,out):
    '''Convert `images' from HWC to CHW format.'''
    return images.transpose([2,0,1]), out


def convertBoth(images,out):
    return images.transpose([2,0,1]), out.transpose([2,0,1])


class NetworkManager(object):
    '''
    This manages the training, loading, saving, and evaluation of a input network. It defines the train, evaluate, and
    infer operations in terms of template methods which must be overridden to suit the specific network being managed.
    Information is loaded, saved, and logged to a directory which the manager creates if it doesn't already exist. The
    method descriptions for netForward(), lossForward(), train(), and evaluate() explain the necessary details to 
    implementing a subtype of this class.
    '''
    def __init__(self,net,opt,loss,isCuda=True,savedirprefix=None,**params):
        '''
        Initialize the manager with the given network `net' to be managed, optimizer `opt', and loss function `loss'.
        If `isCuda' is True the network and inputs are converted to cuda tensors. The `savedirprefix' is the prefix for
        the new directory to create if it doesn't exist, if it does exist it is expected to be a previously created
        directory with a stored network that is reloaded. The `params' value is a user parameter dict for the network.
        '''
        self.net=net
        self.isCuda=isCuda
        self.params=params
        self.opt=opt
        self.loss=loss
        self.traininputs=None
        self.netoutputs=None
        self.lossoutput=None
        
        self.savedir=None
        self.logfilename='train.log'
        
        if isCuda:
            self.net=self.net.cuda()

        if savedirprefix is not None:
            if os.path.exists(savedirprefix):
                self.savedir=savedirprefix
                self.reload()
            else:
                self.savedir='%s-%s'%(savedirprefix,datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                os.mkdir(self.savedir)
                
    def log(self,*items):
        '''Log the given values to self.logfilename is the save directory.'''
        dt=datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S: ')
        msg=dt+' '.join(map(str,items))
        
        if self.savedir:
            with open(os.path.join(self.savedir,self.logfilename),'a') as o:
                print(msg,file=o)
                
    def updateStep(self,step,steploss):
        '''Called after every train step, with arguments for the step number and loss at that step.'''
        pass
    
    def saveStep(self,step,steploss):
        '''Called after every model save operation, with arguments for the step number and loss at that step.'''
        pass
    
    def evalStep(self,index,steploss,results):
        '''
        Called after every evaluation step, with arguments for the step number and loss at that step. The `results' list
        is the accumulated result from each application of this method. Given self.traininputs and self.netoutputs this
        method is expected to calculate some evaluation result or metric from these and append it to `results'.
        '''
        pass
    
    def netForward(self):
        '''
        Called before every train/evaluate step and is expected to apply self.traininputs to the network's forward pass
        and return the results from that. This method must be overridden.
        '''
        pass
    
    def lossForward(self):
        '''
        Called before every train/evaluate step and is expected to use self.netoutputs to calculate a loss value and 
        return the results from that. This method must be overridden.
        '''
        pass
    
    def reload(self):
        '''Reload the network by loading the most recent .pth file in the save directory if there is one.'''
        files=glob.glob(os.path.join(self.savedir,'*.pth'))
        if files:
            self.load(max(files,key=os.path.getctime))
    
    def load(self,path):
        '''Load the network from the given path.'''
        self.net.load_state_dict(torch.load(path))
    
    def save(self,path):
        '''Save the network to the given path.'''
        torch.save(self.net.state_dict(),path)
        
    def convertArray(self,arr):
        '''Convert the Numpy array `arr' to a PyTorch Variable, converting to Cuda if necessary.'''
        arr=torch.autograd.Variable(torch.from_numpy(arr))
        if self.isCuda:
            arr=arr.cuda()
            
        return arr
    
    def toNumpy(self,arr):
        '''Convert the PyTorch Tensor `arr' to a Numpy array.'''
        return arr.cpu().data.numpy()

    def train(self,inputfunc,steps,savesteps=5):
        '''
        Train the network for `step' number of steps starting at 1, saving `savesteps' number of times at regular 
        intervals. The callable `inputfunc' is expected to take no arguments and return a tuple pf batch Numpy arrays of 
        shape, B, BC, BCHW or BCDHW. A train step is composed of these steps:
            1. `inputfunc' is called, each returned value is converted to a Variable, then assigned to self.traininputs
            2. self.netForward() is called and results assigned to self.netoutputs
            3. self.lossForward() is called and results assigned to self.lossoutput
            4. The optimizer performs one training step
            5. self.updateStep() is called and the loss value is assigned to self.params['loss']
            6. If the model is saved on the current step, self.save() is used to save then self.saveStep() is called
            
        Throughout the training process self.log() is called regularly to save logging information.
        '''
        self.log('=================================Starting=================================')
        start=time.time()
        
        try:
            assert self.opt is not None
            assert self.loss is not None
            
            self.log('Params:',self.params)
            self.log('Savedir:',self.savedir)
            
            for s in range(1,steps+1):
                self.log('Timestep',s,'/',steps)
                
                self.traininputs=[self.convertArray(arr) for arr in inputfunc()]                
                self.netoutputs=self.netForward()
            
                self.lossoutput=self.lossForward()
                
                self.opt.zero_grad()
                self.lossoutput.backward()
                self.opt.step()
            
                lossval=toItem(self.lossoutput)
                self.log('Loss:',lossval)
                self.updateStep(s,lossval)
                self.params['loss']=lossval
            
                if self.savedir and (s==steps or (savesteps>0 and (s%(steps//savesteps))==0)):
                    self.save(os.path.join(self.savedir,'net_%.6i.pth'%s))
                    self.saveStep(s,lossval)
                    
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
            
            for i in range(0,inputlen,batchSize):
                self.traininputs=[self.convertArray(arr[i:i+batchSize]) for arr in inputs]
                self.netoutputs=self.netForward()
                self.lossoutput=self.lossForward()
                losses.append(toItem(self.lossoutput))
                
                self.evalStep(i,losses[-1],results)
                # clear stored variables to free graph
                self.traininputs=None
                self.netoutputs=None
                self.lossoutput=None
                
        except Exception as e:
            self.log(e)
            raise
        finally:
            self.log('Total time (s): %s'%(time.time()-start))
            self.log('Losses:',losses)
            self.log('===================================Done===================================')
            
        return losses,results
    
    def infer(self,inputs,batchSize=2):
        '''
        Infer results by applying it to batches of size `batchSize' from the input arrays given in `inputs'. This only 
        uses the forward pass of the network to compute output and does not compute loss or use the optimizer:
            1. Convert each batch slice of arrays in `inputs' to Variables and store all in self.traininputs
            2. self.netForward() is called and results assigned to self.netoutputs
            3. If self.netoutputs is a list or tuple, each tensor it stores is converted to Numpy and appended to the 
            results list, if a single tensor this is converted then appended
            4. Clear the stored variables to free the graph
            
        The result is a list of convert outputs, one for each batch. 
        '''
        assert all(i.shape[0]==inputs[0].shape[0] for i in inputs)
        inputlen=inputs[0].shape[0]
        results=[]
        
        for i in range(0,inputlen,batchSize):
            self.traininputs=[self.convertArray(arr[i:i+batchSize]) for arr in inputs]
            self.netoutputs=self.netForward()
            
            if isinstance(self.netoutputs,(tuple,list)):
                results.append(tuple(map(self.toNumpy, self.netoutputs)))
            else:
                results.append(self.toNumpy(self.netoutputs))
                
            self.traininputs=None
            self.netoutputs=None
        
        return results
    
    
class SegmentMgr(NetworkManager):
    '''
    Basic manager subtype for segmentation, specifying Adam as the optimizer with params['learningRate'] used as
    the learn rate, and a loss function defined as DiceLoss. This expects the first value in self.traininputs to
    be the images and the last to be the masks, and the first value in self.netoutputs to be the logits.
    '''
    def __init__(self,net,isCuda=True,savedirprefix=None,**params):
        opt=torch.optim.Adam(net.parameters(),lr=params.get('learningRate',1e-3))
        loss=params.get('loss',pytorchnet.DiceLoss())
        
        super(SegmentMgr,self).__init__(net,opt,loss,isCuda,savedirprefix,**params)
    
    def netForward(self):
        images=self.traininputs[0]
        return self.net(images)
    
    def lossForward(self):
        masks=self.traininputs[-1]
        logits=self.netoutputs[0]
        return self.loss(logits,masks)    
    
 
class AutoEncoderMgr(NetworkManager):
    '''
    Basic manager subtype for autoencoders, specifying Adam as the optimizer with params['learningRate'] used as
    the learn rate, and a loss function defined as BCEWithLogitsLoss. This expects the first value in self.traininputs 
    to be the input images and the last to be the output images, and the first value in self.netoutputs to be the logits.
    '''
    def __init__(self,net,isCuda=True,savedirprefix=None,loss=None,**params):
        opt=torch.optim.Adam(net.parameters(),lr=params.get('learningRate',1e-3))
        loss=loss if loss is not None else torch.nn.BCEWithLogitsLoss()
        
        super(AutoEncoderMgr,self).__init__(net,opt,loss,isCuda,savedirprefix,**params)
    
    def netForward(self):
        images=self.traininputs[0]
        return self.net(images)
    
    def lossForward(self):
        imgs=self.traininputs[-1]
        logits=self.netoutputs[0]
        return self.loss(logits,imgs)
    
    
class ImageClassifierMgr(NetworkManager):
    '''
    Basic manager subtype for classifier networks, specifying Adam as the optimizer with params['learningRate'] used as
    the learn rate, and a loss function defined as MSELoss. This expects the first value in self.traininputs to be the 
    input images and the last to be the category labels, and the first value in self.netoutputs to be the one-hot vector
    of predictions, ie. of dimensions (batch,# of categories).
    '''
    def __init__(self,net,isCuda=True,savedirprefix=None,loss=None,**params):
        opt=torch.optim.Adam(net.parameters(),lr=params.get('learningRate',1e-3))
        loss=loss if loss is not None else torch.nn.MSELoss()
        
        super(ImageClassifierMgr,self).__init__(net,opt,loss,isCuda,savedirprefix,**params)
        
    def netForward(self):
        images=self.traininputs[0]
        return self.net(images)
    
    def lossForward(self):
        values=self.traininputs[-1]
        preds=self.netoutputs[0]
        return self.loss(preds,values)
    