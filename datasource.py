
from __future__ import division, print_function
import threading
import multiprocessing as mp
import platform
from multiprocessing import sharedctypes
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
    
    def getData(batchSize=None,selectProbs=None,chosenInds=None):
        if chosenInds is None:
            chosenInds=np.random.choice(inShape[0],batchSize,p=selectProbs)
        
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
        
        
def fillArraySet(arrays,writes):
    if isinstance(arrays,tuple):
        for a,w in zip(arrays,writes):
            a[:]=w
    else:
        arrays[:]=writes
        
        
def toShared(arrays):
    '''Convert the given Numpy array to a shared ctypes object.'''
    if isinstance(arrays,tuple):
        out=[]
        for arr in arrays:
            carr=np.ctypeslib.as_ctypes(arr)
            out.append(sharedctypes.RawArray(carr._type_, carr))
            
        return tuple(out)
    else:
        carr=np.ctypeslib.as_ctypes(arrays)
        return sharedctypes.RawArray(carr._type_, carr)


def fromShared(arrays):
    '''Map the given ctypes object to a Numpy array, this is expected to be a shared object from the parent.'''
    if isinstance(arrays,tuple):
        return tuple(np.ctypeslib.as_array(arr) for arr in arrays)
    else:
        return np.ctypeslib.as_array(arrays)

        
def init(inArrays_,outArrays_,inAugs_,outAugs_,augments_):
    global inArrays
    global outArrays
    global inAugs
    global outAugs
    global augments
    
    inArrays=fromShared(inArrays_)
    outArrays=fromShared(outArrays_)
    inAugs=fromShared(inAugs_)
    outAugs=fromShared(outAugs_)
    
    augments=augments_
    
    
def applyAugments(indices):
    global inArrays
    global outArrays
    global inAugs
    global outAugs
    global augments
    
    for i in indices:
        ina=getArraySet(inArrays,i)
        outa=getArraySet(outArrays,i)
        
        for aug in augments:
            ina,outa=aug(ina,outa)
        
        writeArraySet(inAugs,ina,i)
        writeArraySet(outAugs,outa,i)
        
    
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
    def threadBatchGen(self,batchSize,numThreads=None):
        numThreads=min(batchSize,numThreads or mp.cpu_count())
        threadIndices=np.array_split(np.arange(batchSize),numThreads)
        isRunning=True
        batchQueue=queue.Queue(1)
        
        inArrays,outArrays=self.getIndexBatch([0])
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
            try:
                batchQueue.get(False) # there may be a batch waiting on the queue, batchThread is stuck until this is removed
            except queue.Empty:
                pass
            
    @contextmanager
    def processBatchGen(self,batchSize,numProcs=None):
        assert platform.system().lower()!='windows', 'Generating batches with processes requires fork() semantics not present in Windows.'
        
        numProcs=min(batchSize,numProcs or mp.cpu_count())
        procIndices=np.array_split(np.arange(batchSize),numProcs)
        isRunning=True
        batchQueue=queue.Queue(1)
        
        inArrays,outArrays=self.getRandomBatch(batchSize)
        inAugTest,outAugTest=self.applyAugments(getArraySet(inArrays,0),getArraySet(outArrays,0))
        
        inAugs=createZeroArraySet(inAugTest,(batchSize,))
        outAugs=createZeroArraySet(outAugTest,(batchSize,))
        
        # convert original and augmented arrays to shared arrays
        inArrays=toShared(inArrays)
        outArrays=toShared(outArrays)
        inAugs=toShared(inAugs)
        outAugs=toShared(outAugs)
        
        maugs=self.augments
        initargs=(inArrays,outArrays,inAugs,outAugs,maugs)
               
        def _batchThread(inArrays,outArrays,inAugs,outAugs,maugs):
            try:
                initargs=(inArrays,outArrays,inAugs,outAugs,maugs)
                
                with mp.Pool(numProcs,initializer=init,initargs=initargs) as p:
                    inArrays=fromShared(inArrays)
                    outArrays=fromShared(outArrays)
                    inAugs=fromShared(inAugs)
                    outAugs=fromShared(outAugs)
                        
                    while isRunning:
                        inArraysb,outArraysb=self.getRandomBatch(batchSize)
                        fillArraySet(inArrays,inArraysb)
                        fillArraySet(outArrays,outArraysb)
                        
                        if maugs:
                            p.map(applyAugments,procIndices)

                        batchQueue.put((inAugs,outAugs))
                        
            except Exception as e:
                batchQueue.put(e)
                
        batchThread=threading.Thread(target=_batchThread,args=initargs)
        batchThread.start()
        
        def _get():
            v=batchQueue.get()
            if not isinstance(v,tuple):
                raise v
                
            return v
        
        try:
            yield _get
        finally:
            isRunning=False
            try:
                batchQueue.get(False) # there may be a batch waiting on the queue, batchThread is stuck until this is removed
            except queue.Empty:
                pass
        
        
        