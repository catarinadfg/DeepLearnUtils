
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
        
        
def initProc(inArrays_,augs_,augments_):
    '''Initialize subprocesses by setting global variables.'''
    global inArrays
    global augs
    global augments
    inArrays=fromShared(inArrays_)
    augs=fromShared(augs_)
    augments=augments_
    
    
def applyAugmentsProc(indices):
    '''Apply the augmentations to the input array at the given indices.'''
    global inArrays
    global augs
    global augments
    
    for i in indices:
        inarrs=[a[i] for a in inArrays]
        
        for aug in augments:
            inarrs=aug(*inarrs)
            
        for ina,outa in zip(inarrs,augs):
            outa[i]=ina
        

class DataSource(object):
    def __init__(self,*arrays,dataGen=None,selectProbs=None,augments=[]):
        self.arrays=list(arrays)
        self.dataGen=dataGen or self.defaultDataGen
        self.selectProbs=selectProbs
        self.augments=augments
        
    def defaultDataGen(self,batchSize=None,selectProbs=None,chosenInds=None):
        if chosenInds is None:
            chosenInds=np.random.choice(self.arrays[0].shape[0],batchSize,p=selectProbs)
                
        return tuple(a[chosenInds] for a in self.arrays)
                
    def getRandomBatch(self,batchSize):
        '''Call the generator callable with the given `batchSize' value with self.selectProb as the second argument.'''
        return self.dataGen(batchSize,self.selectProbs)
    
    def getIndexBatch(self,chosenInds):
        '''Call the generator callable with `chosenInds' as the chosen indices to select values from.'''
        return self.dataGen(chosenInds=chosenInds)
    
    def getAugmentedArrays(self,arrays):
        '''Apply the augmentations to single-instance arrays.'''
        for aug in self.augments:
            arrays=aug(*arrays)
            
        return arrays
    
    def applyAugments(self,arrays,augArrays,indices=None):
        '''Apply the augmentations to batch input and output arrays at `indices' or for the whole arrays if not given.'''
        indices=range(arrays[0].shape[0]) if indices is None else indices
            
        for i in indices:
            inarrs=[a[i] for a in arrays]
            outarrs=self.getAugmentedArrays(inarrs)
            for out,aug in zip(outarrs,augArrays):
                aug[i]=out
                
    @contextmanager
    def threadBatchGen(self,batchSize,numThreads=None):
        '''Yields a callable object which produces `batchSize' batches generated in `numThreads' threads.'''
        numThreads=min(batchSize,numThreads or mp.cpu_count())
        threadIndices=np.array_split(np.arange(batchSize),numThreads)
        isRunning=True
        batchQueue=queue.Queue(1)
        
        inArrays=self.getIndexBatch([0])
        augTest=self.getAugmentedArrays([a[0] for a in inArrays])
        
        augs=tuple(np.zeros((batchSize,)+a.shape,a.dtype) for a in augTest)
        
        def _batchThread():
            while isRunning:
                threads=[]
                batch=self.getRandomBatch(batchSize)
                
                for indices in threadIndices:
                    t=threading.Thread(target=self.applyAugments,args=(batch,augs,indices))
                    t.start()
                    threads.append(t)
                    
                for t in threads:
                    t.join()
                    
                batchQueue.put(tuple(a.copy() for a in augs))
                
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
        '''Yields a callable object which produces `batchSize' batches generated in `numProcs' subprocesses.'''
        assert platform.system().lower()!='windows', 'Generating batches with processes requires fork() semantics not present in Windows.'
        
        numProcs=min(batchSize,numProcs or mp.cpu_count())
        procIndices=np.array_split(np.arange(batchSize),numProcs)
        isRunning=True
        batchQueue=mp.Queue(1)
        
        inArrays=self.getRandomBatch(batchSize)
        augTest=self.getAugmentedArrays([a[0] for a in inArrays])
        
        augs=tuple(toShared(np.zeros((batchSize,)+a.shape,a.dtype)) for a in augTest)
        
        inArrays=tuple(map(toShared,inArrays))
        
        maugs=self.augments
        initargs=(inArrays,augs,maugs)
               
        def _batchThread(inArrays,augs,maugs):
            try:
                initargs=(inArrays,augs,maugs)
                
                with mp.Pool(numProcs,initializer=initProc,initargs=initargs) as p:
                    inArrays=tuple(map(fromShared,inArrays))
                    augs=tuple(map(fromShared,augs))
                        
                    while isRunning:
                        batch=self.getRandomBatch(batchSize)
                        for a,b in zip(inArrays,batch):
                            a[...]=b
                            
                        if maugs:
                            p.map(applyAugmentsProc,procIndices)

                        batchQueue.put(tuple(a.copy() for a in augs))
                        
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
        
        
def randomDataSource(shape,augments=[],dtype=np.float32):
    '''
    Returns a DataSource producing batches of `shape'-sized standard normal random arrays of type `dtype'. The `augments'
    list of augmentations is pass to the DataSource object when constructed. The input and output are the same array.
    '''
    def randData(batchSize=None,selectProbs=None,chosenInds=None):
        if chosenInds: # there are no arrays to index from so use the list size as batchSize instead
            batchSize=len(chosenInds)

        randvals=np.random.randn(batchSize, *shape).astype(dtype)
        return randvals,randvals
    
    return DataSource(dataGen=randData,augments=augments)

        
class BufferDataSource(DataSource):
    def appendBuffer(self,*arrays):
        if not self.arrays:
            self.arrays=list(arrays)
        else:
            for i in range(len(self.arrays)):
                self.arrays[i]=np.concatenate([self.arrays[i],arrays[i]])
                
        if self.selectProbs is not None:
            self.selectProbs=np.ones((self.arrays[0].shape[0],))
            
    def clearBuffer(self):
        self.arrays=[]
        
        if self.selectProbs is not None:
            self.selectProbs=self.selectProbs[:0]
        

if __name__=='__main__':
    def testAug(im,cat):
        return im[0],cat

    src=DataSource(np.random.randn(10,1,16,16),np.random.randn(10,2),augments=[testAug])
    
    with src.processBatchGen(4) as gen:
        batch=gen()
        
        print([a.shape for a in batch])
        
    bsrc=BufferDataSource()#np.random.randn(0,1,16,16),np.random.randn(0,2))
    
    bsrc.appendBuffer(np.random.randn(10,1,16,16),np.random.randn(10,2))
    
    with bsrc.processBatchGen(4) as gen:
        batch=gen()
        print([a.shape for a in batch])
    
    bsrc.clearBuffer()
        
    with bsrc.processBatchGen(4) as gen:
        batch=gen()
        print([a.shape for a in batch])
        