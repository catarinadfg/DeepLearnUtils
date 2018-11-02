
from __future__ import division, print_function
import threading
import multiprocessing
from contextlib import contextmanager

try:
    import queue
except:
    import Queue as queue

import numpy as np


def createDataGenerator(inArrays,outArrays):
    if isinstance(inArrays,tuple):
        inFunc=lambda i:tuple(a[i] for a in inArrays)
        
        inShape=inArrays[0].shape
        assert all(a.shape==inShape for a in inArrays)
    else:
        inFunc=lambda i:inArrays[i]
        inShape=inArrays.shape
        
    if isinstance(outArrays,tuple):
        outFunc=lambda i:tuple(a[i] for a in outArrays)
        
        outShape=outArrays[0].shape
        assert all(a.shape==outShape for a in outArrays)
    else:
        outFunc=lambda i:outArrays[i]
        outShape=outArrays.shape
        
    assert inShape[0]==outShape[0]
    indices=list(range(inShape[0]))
    
    def getData(batchSize=None,selectProbs=None,chosenInds=None):
        chosenInds=chosenInds or np.random.choice(indices,batchSize,p=selectProbs)
        
        return inFunc(chosenInds),outFunc(chosenInds)
    
    return getData


def createZeroArraySet(arrays,addShape):
    if isinstance(arrays,tuple):
        return tuple(np.zeros(addShape+a.shape,a.dtype) for a in arrays)
    else:
        return np.zeros(addShape+arrays.shape,arrays.dtype)
    

def getArraySet(arrays,indices):
    return tuple(a[indices] for a in arrays) if isinstance(arrays,tuple) else arrays[indices]


def writeArraySet(arrays,writes,index):
    if isinstance(arrays,tuple):
        for a,w in zip(arrays,writes):
            a[index]=w
    else:
        arrays[index]=writes
        

class DataSource(object):
    def __init__(self,inArrays=None,outArrays=None,dataGen=None,selectProbs=None,augments=[]):
        self.dataGen=dataGen or createDataGenerator(inArrays,outArrays)
        self.selectProbs=selectProbs
        self.augments=augments
        
    def getRandomBatch(self,batchSize):
        return self.dataGen(batchSize,self.selectProbs)
    
    def getIndexBatch(self,chosenInds):
        return self.dataGen(chosenInds=chosenInds)
    
    def applyAugments(self,inArrays,outArrays):
        for aug in self.augments:
            inArrays,outArrays=aug(inArrays,outArrays)
            
        return inArrays,outArrays
    
    @contextmanager
    def asyncBatchGen(self,batchSize,numThreads=None):
        numThreads=min(batchSize,numThreads or multiprocessing.cpu_count())
        threadIndices=np.array_split(np.arange(batchSize),numThreads)
        isRunning=True
        batchQueue=queue.Queue(1)
        
        inArrays,outArrays=self.getRandomBatch(1)
        inAugTest,outAugTest=self.applyAugments(getArraySet(inArrays,0),getArraySet(outArrays,0))
        
        inAugs=createZeroArraySet(inAugTest,(batchSize,))
        outAugs=createZeroArraySet(outAugTest,(batchSize,))
        
        def _batchThread():
            while isRunning:
                threads=[]
                inArrays,outArrays=self.getRandomBatch(batchSize)
                
                def _generateForIndices(indices):
                    for i in indices:
                        ina,outa=self.applyAugments(getArraySet(inArrays,i),getArraySet(outArrays,i))
                        writeArraySet(inAugs,ina,i)
                        writeArraySet(outAugs,outa,i)
                
                for indices in threadIndices:
                    t=threading.Thread(target=_generateForIndices,args=(indices,))
                    t.start()
                    threads.append(t)
                    
                for t in threads:
                    t.join()
                    
                batchQueue.put((inAugs,outAugs))
                
                
                
        batchThread=threading.Thread(target=_batchThread)
        batchThread.start()
        
        try:
            yield batchQueue.get
        finally:
            isRunning=False
        
        
        
        
        