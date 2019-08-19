# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

from functools import wraps
from multiprocessing.pool import ThreadPool
from queue import Queue, Full, Empty
from threading import Thread, Event, RLock
from contextlib import contextmanager
import numpy as np


class OrderType(object):
    SHUFFLE='shuffle'
    CHOICE='choice'
    LINEAR='linear'
    

class DataStream(object):
    """
    The DataStream class represents a chain of iterable objects where one iterates over its source and
    in turn yields values which are possibly transformed. This allows an intermediate object in the
    stream to modify a data element which passes through the stream or generate more than out output
    value for each input. The type relies on an input source which must be an iterable, then passes
    each item from the source into the generate() generator method to produce one or more items which
    it then yields. A sequence of stream objects is created by using one stream as the source to another.
    Subclasses can override generate() to produce filter or transformer types to place in a sequence 
    DataStream objects, or use the `streamgen` decorator to do the same. 
    """
    def __init__(self,src):
        """Initialize with `src' as the source iterable, and self.isRunning as True."""
        self.src=src
        self.isRunning=True
        
    def __iter__(self):
        """
        Iterate over every value from self.src, passing through self.generate() and yielding the
        values it generates.
        """
        for srcVal in self.src:
            for outVal in self.generate(srcVal):
                yield outVal # yield with syntax too new?
                
    def generate(self,val):
        """Generate values from input `val`, by default just yields that. """
        yield val
        
    def stop(self):
        """Sets self.isRunning to False and calls stop() on self.src if it has this method."""
        self.isRunning=False
        if hasattr(self.src,'stop'):
            self.src.stop()
            
    def getGenFunc(self):
        stream=iter(self)
        return lambda:next(stream)
        
    
class FuncStream(DataStream):
    """For use with `streamgen`, the given callable is used as the generator in place of generate()."""
    def __init__(self,src,func,fargs,fkwargs):
        super().__init__(src)
        self.func=func
        self.fargs=fargs
        self.fkwargs=fkwargs
        
    def generate(self,val):
        for outVal in self.func(val,*self.fargs,**self.fkwargs):
            yield outVal
            
    
def streamgen(func):
    """
    Converts a generator function into a constructor for creating FuncStream instances 
    using the function as the generator.
    """
    @wraps(func)
    def _wrapper(src,*args,**kwargs):
        return FuncStream(src,func,args,kwargs)
    
    return _wrapper
        

class ArraySource(DataStream):
    """
    Creates a data source from one or more equal length arrays. Each data item yielded is a tuple of slices
    containing a single index in the 0th dimension (ie. batch dimension) for each array. By default values
    are drawn in sequential order but can be set to shuffle the order so that each value appears exactly once
    per epoch, or to choose a random selection which may include items multiple times or not at all based off
    an optional probability distribution. By default the stream will iterate over the arrays indefinitely or
    optionally only once.
    """
    def __init__(self,*arrays,orderType=OrderType.LINEAR,doOnce=False,choiceProbs=None):
        self.arrays=tuple(map(np.atleast_1d,arrays))
        arrayLen=self.arrays[0].shape[0]
        
        if any(arr.shape[0]!=arrayLen for arr in self.arrays):
            raise ValueError('All input arrays must have the same length for dimension 0')
            
        if orderType not in (OrderType.SHUFFLE,OrderType.CHOICE,OrderType.LINEAR):
            raise ValueError('Invalid orderType value %r'%(orderType,))
        
        self.orderType=orderType
        self.doOnce=doOnce
        self.choiceProbs=None
        
        if self.choiceProbs is not None:
            if self.choiceProbs.shape[0]!=arrayLen:
                raise ValueError('Length of choiceProbs (%i) must match that of input arrays (%i)'%
                                 (self.choiceProbs.shape[0],arrayLen))
                
            self.choiceProbs=np.atleast_1d(self.choiceProbs)/np.sum(self.choiceProbs)
        
        super().__init__(self.yieldArrays())
        
    def yieldArrays(self):
        arrayLen=self.arrays[0].shape[0]
        indices=np.arange(arrayLen)
        
        while self.isRunning:
            if self.orderType==OrderType.SHUFFLE:
                np.random.shuffle(indices)
            elif self.orderType==OrderType.CHOICE:
                indices=np.random.choice(range(arrayLen),arrayLen,p=self.choiceProbs)
                
            for i in indices:
                yield tuple(arr[i] for arr in self.arrays)
                
            if self.doOnce:
                break
                
    def getSubArrays(self,indices):
        subArrays=[a[indices] for a in self.arrays]
        subProbs=None
        
        if self.choiceProbs is not None:
            subProbs=self.choiceProbs[indices]
            subProbs=subProbs/np.sum(subProbs)
            
        return ArraySource(*subArrays,orderType=self.orderType,doOnce=self.doOnce,choiceProbs=subProbs)
                
                
class NPZFileSource(ArraySource):
    def __init__(self,fileName,arrayNames,otherValues=[],orderType=OrderType.LINEAR,doOnce=False):
        self.fileName=fileName
        
        dat=np.load(fileName)
        
        keys=set(dat.keys())
        missing=set(arrayNames)-keys
        
        if missing:
            raise ValueError('Array name(s) %r not in loaded npz file'%(missing,))
                
        arrays=[dat[name] for name in arrayNames]
        
        super().__init__(*arrays,orderType=orderType,doOnce=doOnce)
        
        self.otherValues={n:dat[n] for n in otherValues if n in keys}
        

class RandomGenerator(DataStream):
    """Randomly generate float32 arrays of the given shape."""
    def __init__(self,*shape):
        super().__init__(self.generateRandArray(shape))

    def generateRandArray(self,shape):
        while self.isRunning:
            yield np.random.rand(*shape)
    
    
class TestImageGenerator(DataStream):
    """Generate 2D image/seg test image pairs."""
    def __init__(self,width,height,numObjs=12,radMax=30,noiseMax=0.0,numSegClasses=5):
        self.doGen=True
        self.width=width
        self.height=height
        self.numObjs=numObjs
        self.radMax=radMax
        self.noiseMax=noiseMax
        self.numSegClasses=numSegClasses
        
        from trainutils import createTestImage
        self.func=createTestImage
        
        super().__init__(self.generateImage())
        
    def generateImage(self):
        while self.isRunning:
            yield self.func(self.width,self.height,self.numObjs,self.radMax,self.noiseMax,self.numSegClasses)
            

class BatchStream(DataStream):
    def __init__(self,src,batchSize):
        super().__init__(src)
        self.batchSize=batchSize
        
    def __iter__(self):
        srcVals=[]
        
        for srcVal in super().__iter__():
            srcVals.append(srcVal)

            if len(srcVals)==self.batchSize:
                yield tuple(map(np.stack,zip(*srcVals)))
                srcVals=[]
                
            
class AugmentStream(BatchStream):
    def __init__(self,src,batchSize,augments=[]):
        super().__init__(src,batchSize)
        self.augments=augments
        
    def generate(self,arrays):
        '''Apply the augmentations to single-instance arrays, yielding a single set of arrays.'''
        for aug in self.augments:
            arrays=aug(*arrays)
            
        yield arrays
        

class ThreadAugmentStream(AugmentStream):
    def __init__(self,src,batchSize,numThreads=None,augments=[]):
        super().__init__(src,batchSize,augments)
        self.numThreads=numThreads
        self.batchArrays=None
        
    def getLocalGen(self):
        '''Returns a non-threaded iterator, ie. behaves like AugmentStream.'''
        return super().__iter__()
        
    def applyAugments(self,arrays):
        '''Apply the augmentations to single-instance arrays, returning a single set of arrays.'''
        for a in self.generate(arrays):
            return a
        
    def _applyAugmentThread(self,index,arrays):
        '''
        Apply the augmentations to `arrays` and storing results in the position `index` in the appropriate array of 
        self.batchArrays. This is meant to be called by threads.
        '''
        arrays=self.applyAugments(arrays)
        
        for i,arr in enumerate(arrays):
            self.batchArrays[i][index][...]=arr
        
    def __iter__(self):
        srcVals=[]
        arraySizeTypes=None
        
        with ThreadPool(self.numThreads) as tp:
            for srcVal in self.src:
                srcVals.append(srcVal)

                if arraySizeTypes is None:
                    testAug=self.applyAugments(srcVals[0])
                    arraySizeTypes=tuple(((self.batchSize,)+a.shape,a.dtype) for a in testAug)
                    
                if len(srcVals)==self.batchSize:
                    self.batchArrays=tuple(np.zeros(*st) for st in arraySizeTypes) # create fresh arrays each time
                    tp.starmap(self._applyAugmentThread,list(enumerate(srcVals)))
                    yield self.batchArrays
                    srcVals=[]
                            
                        
class ThreadBufferStream(DataStream):
    def __init__(self,src,bufferSize=1,timeout=0.01):
        super().__init__(src)
        self.bufferSize=bufferSize
        self.timeout=timeout
        self.rlock=RLock()
        self.buffer=Queue(self.bufferSize)
        self.stopEvent=Event()
        
    def enqueueValues(self):
        # allows generate() to be overridden and used here (instead of iter(self.src))
        for srcVal in super().__iter__():
            while self.isRunning and not self.stopEvent.is_set():
                try:
                    self.buffer.put(srcVal,timeout=self.timeout)
                except Full:
                    pass # try to add the item again
                else:
                    break # successfully added the item, quit trying
             
            if self.stopEvent.is_set(): # quit the thread cleanly when the event is set
                return
            
    def __enter__(self):
        if not self.rlock.acquire(False):
            raise ValueError('Cannot acquire thread lock for this stream')

        try:
            self.stopEvent.clear()
            return self
        except:
            self.rlock.release()
            
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.rlock.acquire(False):
            raise ValueError('Cannot acquire thread lock for this stream')
            
        self.stopEvent.set()
        self.rlock.release()
        
    def __iter__(self):
        if not self.rlock.acquire(False):
            raise ValueError('Cannot acquire thread lock for this stream')
            
        self.stopEvent.clear()
        genThread=Thread(target=self.enqueueValues,daemon=True)
        genThread.start()
        
        try:
            while self.isRunning and genThread.is_alive():
                try:
                    yield self.buffer.get(timeout=self.timeout)
                except Empty:
                    pass # queue was empty this time, try again
        finally:
            self.stopEvent.set()   
            genThread.join()
            self.rlock.release()
        
#    def __iter__(self):
#        buffer=Queue(self.bufferSize)
#        stopEvent=Event()
#        
#        genThread=Thread(target=self.enqueueValues,args=(buffer,stopEvent),daemon=True)
#        genThread.start()
#        
#        try:
#            while self.isRunning and genThread.is_alive():
#                try:
#                    yield buffer.get(timeout=self.timeout)
#                except Empty:
#                    pass # queue was empty this time, try again
#        finally:
#            stopEvent.set()   
#            genThread.join()
            
#    @contextmanager
#    def iterAsync(self):
#        buffer=Queue(self.bufferSize)
#        stopEvent=Event()
#        
#        genThread=Thread(target=self.enqueueValues,args=(buffer,stopEvent),daemon=True)
#        genThread.start()
#            
#        def _iterBuffer():
#            try:
#                while self.isRunning and genThread.is_alive():
#                    try:
#                        yield buffer.get(timeout=self.timeout)
#                    except Empty:
#                        pass # queue was empty this time, try again
#            finally:
#                stopEvent.set()   
#
#        try:
#            yield _iterBuffer
#        finally:
#            stopEvent.set()
#            genThread.join()
        