
from __future__ import print_function,division
import os
import glob
import datetime
import torch
import pytorchnet
import numpy as np


def convertAug(images,masks,isCuda=True):
    images=images.transpose([2,0,1])
    masks=masks[np.newaxis,:,:]
        
    return images,masks


class NetworkManager(object):
    def __init__(self,net,opt,loss,isCuda=True,savedirprefix=None,params={}):
        self.net=net
        self.isCuda=isCuda
        self.params=params
        self.opt=opt
        self.loss=loss
        self.traininputs=None
        self.netoutputs=None
        
        self.savedir=None
        self.logfilename='train.log'
        
        if isCuda:
            self.net=self.net.cuda()

        if savedirprefix:
            if os.path.exists(savedirprefix):
                self.savedir=savedirprefix
            else:
                self.savedir='%s-%s'%(savedirprefix,datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                os.mkdir(self.savedir)
                
    def log(self,*items):
        dt=datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S: ')
        msg=dt+' '.join(map(str,items))
        
        if self.savedir:
            with open(os.path.join(self.savedir,self.logfilename),'a') as o:
                print(msg,file=o)
                
    def updateStep(self,step,steploss):
        pass
    
    def netForward(self):
        pass
    
    def lossForward(self):
        pass
    
    def reload(self):
        files=glob.glob(os.path.join(self.savedir,'*.pkl'))
        if files:
            self.load(max(files,key=os.path.getctime))
    
    def load(self,path):
        self.net.load_state_dict(torch.load(path))
    
    def save(self,path):
        torch.save(self.net.state_dict(),path)
        
    def convertArray(self,arr):
        arr=torch.autograd.Variable(torch.from_numpy(arr))
        if self.isCuda:
            arr=arr.cuda()
            
        return arr

    def train(self,inputfunc,steps,savesteps=5):
        self.log('===================================Starting===================================')
        
        try:
            assert self.opt is not None
            assert self.loss is not None
            
            self.log('Params:',self.params)
            self.log('Savedir:',self.savedir)
            
            for s in range(steps):
                self.log('Timestep',s,'/',steps)
                
                self.traininputs=[self.convertArray(arr) for arr in inputfunc()]                
                self.netoutputs=self.netForward()
            
                loss=self.lossForward()
                
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            
                self.log('Loss:',loss.data[0])
                self.updateStep(s,loss.data[0])
            
                if self.savedir and savesteps>0 and ((s+1)%(steps//savesteps))==0:
                    self.save(os.path.join(self.savedir,'net_%.5i.pkl'%s))
                    
        except Exception as e:
            self.log(e)
            raise
        finally:
            self.log('===================================Done===================================')

    def evaluate(self,inputs):
        self.traininputs=[self.convertArray(arr) for arr in inputs]
        self.netoutputs=self.netForward()
        loss=self.lossForward()
        return loss.data[0]
        
    
class BinarySegmentMgr(NetworkManager):
    def __init__(self,net,isCuda=True,savedirprefix=None,params={}):
        opt=torch.optim.Adam(net.parameters(),lr=params['learningRate'])
        loss=pytorchnet.BinaryDiceLoss()
        
        super(BinarySegmentMgr,self).__init__(net,opt,loss,isCuda,savedirprefix,params)
    
    def netForward(self):
        images,_=self.traininputs
        return self.net(images)
    
    def lossForward(self):
        _,masks=self.traininputs
        logits,_=self.netoutputs
        return self.loss(logits,masks)
    