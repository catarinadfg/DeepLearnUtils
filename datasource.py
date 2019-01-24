
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
        

def copyArraySet(arrays):
    if isinstance(arrays,tuple):
        return tuple(a.copy() for a in arrays)
    else:
        return arrays.copy()
        
        
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

        
def initProc(inArrays_,outArrays_,inAugs_,outAugs_,augments_):
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
    
    
def applyAugmentsProc(indices):
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
    '''
    Class for generating data batches from array data. It takes as input arrays of input and output data (of tuples of
    arrays) or a data generation function. Whenever a batch is requested these are used to create a batch of images which
    have been aplied to a list of augmentation functions to modify the data. Two context manager methods are provided for
    generating these batches in separate threads or separate processes. 
    '''
    def __init__(self,inArrays=None,outArrays=None,dataGen=None,selectProbs=None,augments=[]):
        '''
        Initialize the source with either data arrays or a data generation function. If `dataGen' is None then `inArrays'
        and `outArrays' must be provided and be numpy arrays or tuples thereof, createDataGenerator() is used to create
        the generator function with these as input. If `selectProbs' is given this is the normalized probabilities vector
        defining the likelihood an entry from `inArrays' and `outArrays' is selected when making a randomized batch.
        
        The `augments' list is the list of augment callables which batches are passed through before being returned. 
        Each such callable is expected to take as input two numpy arrays (or tuples thereof) each containing a single 
        instance of a data value (eg. a single image). All arrays in `inArrays' and `outArrays' are expected to be in 
        B[H][W][D]C ordering, that is batch is first dimension and channels last.
        
        If `dataGen' is provided it must be a callable accepting 3 arguments (batchSize, selectProbs, chosenInds):
          -If batchSize is given the callable should return a pair of numpy arrays (or tuples thereof) with batchSize 
           number of entries or greater. If `selectProbs' is given this is the selecting probability for each value 
           in the source arrays from which random selections are taken. 
        
          -If chosenInds is provided then these indices from the source arrays are returned instead rather than batchSize 
           number of random selections; in this case batchSize and selectProbs are ignored.
           
        This generator callable is expected to return a pair of numpy arrays or tuples thereof. The `inArrays' and 
        `outArrays' arguments are not used if `dataGen' is provided but the `selectProbs' argument of this constructor 
        is passsed as the `selectProbs' argument whenever the generator is called.
        '''
        self.dataGen=dataGen or createDataGenerator(inArrays,outArrays)
        self.selectProbs=selectProbs
        self.augments=augments
        
    def getRandomBatch(self,batchSize):
        '''Call the generator callable with the given `batchSize' value with self.selectProb as the second argument.'''
        return self.dataGen(batchSize,self.selectProbs)
    
    def getIndexBatch(self,chosenInds):
        '''Call the generator callable with `chosenInds' as the chosen indices to select values from.'''
        return self.dataGen(chosenInds=chosenInds)
    
    def getAugmentedArrays(self,inArrays,outArrays):
        '''Apply the augmentations to single-instance input and output arrays.'''
        for aug in self.augments:
            inArrays,outArrays=aug(inArrays,outArrays)
            
        return inArrays,outArrays
    
    def applyAugments(self,inArrays,outArrays,inAugArrays,outAugArrays,indices=None):
        '''Apply the augmentations to batch input and output arrays at `indices' or for the whole arrays if not given.'''
        if indices is None:
            arr=inArrays[0] if isinstance(inArrays,tuple) else inArrays # assumes all arrays are the same length
            indices=range(arr.shape[0])
            
        for i in indices:
            ina,outa=self.getAugmentedArrays(getArraySet(inArrays,i),getArraySet(outArrays,i))
            writeArraySet(inAugArrays,ina,i)
            writeArraySet(outAugArrays,outa,i)
    
    @contextmanager
    def threadBatchGen(self,batchSize,numThreads=None):
        '''Yields a callable object which produces `batchSize' batches generated in `numThreads' threads.'''
        numThreads=min(batchSize,numThreads or mp.cpu_count())
        threadIndices=np.array_split(np.arange(batchSize),numThreads)
        isRunning=True
        batchQueue=queue.Queue(1)
        
        inArrays,outArrays=self.getIndexBatch([0])
        inAugTest,outAugTest=self.getAugmentedArrays(getArraySet(inArrays,0),getArraySet(outArrays,0))
        
        inAugs=createZeroArraySet(inAugTest,(batchSize,))
        outAugs=createZeroArraySet(outAugTest,(batchSize,))
        
        def _batchThread():
            while isRunning:
                threads=[]
                inArrays,outArrays=self.getRandomBatch(batchSize)
                
                for indices in threadIndices:
                    t=threading.Thread(target=self.applyAugments,args=(inArrays,outArrays,inAugs,outAugs,indices))
                    t.start()
                    threads.append(t)
                    
                for t in threads:
                    t.join()
                    
                batchQueue.put((copyArraySet(inAugs),copyArraySet(outAugs)))
                
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
        
        inArrays,outArrays=self.getRandomBatch(batchSize)
        inAugTest,outAugTest=self.getAugmentedArrays(getArraySet(inArrays,0),getArraySet(outArrays,0))
        
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
                
                with mp.Pool(numProcs,initializer=initProc,initargs=initargs) as p:
                    inArrays=fromShared(inArrays)
                    outArrays=fromShared(outArrays)
                    inAugs=fromShared(inAugs)
                    outAugs=fromShared(outAugs)
                        
                    while isRunning:
                        inArraysb,outArraysb=self.getRandomBatch(batchSize)
                        fillArraySet(inArrays,inArraysb)
                        fillArraySet(outArrays,outArraysb)
                        
                        if maugs:
                            p.map(applyAugmentsProc,procIndices)

                        batchQueue.put((copyArraySet(inAugs),copyArraySet(outAugs)))
                        
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

        