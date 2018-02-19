
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import os
import datetime
import collections
import threading

try:
    import queue
except:
    import Queue as queue
    

def averageGradients(tower_grads):
    '''
    Given the gradients `towergrads' from each computation tower, return the averaged gradient for each variable.
    This is a condensed version of the function in the tutorials on multi-GPU training.
    '''
    avggrads = []
    # each grad_and_vars looks like ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    for grad_and_vars in zip(*tower_grads):
        grads=[tf.expand_dims(g, 0) for g,_ in grad_and_vars if g is not None]
        assert len(grads)>0, 'No variables have gradients'
        
        grad=tf.concat(grads,0)
        grad=tf.reduce_mean(grad, 0) 

        # variables are shared across towers, need only return first tower's variable refs
        avggrads.append((grad,grad_and_vars[0][1]))
    return avggrads


def binaryMaskDiceLoss(logits, labels, smooth=1e-5):
    '''Return the binary mask dice loss between the given logits and labels.'''
    axis = list(range(1, logits.shape.ndims - 1))
    logits=tf.cast(logits,tf.float32)
    labels=tf.cast(labels,tf.float32)

    probs = tf.nn.sigmoid(logits)[..., 0]
    label_sum = tf.reduce_sum(labels, axis=axis, name='label_sum')
    pred_sum = tf.reduce_sum(probs, axis=axis, name='pred_sum')
    intersection = tf.reduce_sum(labels * probs, axis=axis, name='intersection')
    sums=label_sum + pred_sum

    dice = tf.reduce_mean((2.0 * intersection + smooth) / (sums + smooth))
    return 1.0-dice


class GraphImageHook(tf.train.SessionRunHook):
    '''
    This hook keeps track of the nominated scalar and image values. This is used to output graphed histories of the scalars
    with the images during training. For each named scalar value a history of values is stored along with a rolling average.    
    '''
    def __init__(self, fetches,graphnames,imagenames):
        self.fetches = fetches
        self.graphnames=graphnames
        self.imagenames=imagenames
        self.avgLength=50
        self.graphvalues=collections.OrderedDict([(n,[]) for n in graphnames]+[(n+' Avg',[]) for n in graphnames])
        self.images={}
        
    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self.fetches)
    
    def after_run(self, run_context, run_values):
        res=run_values.results
        
        for n in self.graphnames:
            v=res[n]
            self.graphvalues[n].append(v)
            rollingAvg=np.average(self.graphvalues[n][-self.avgLength:])
            self.graphvalues[n+' Avg'].append(rollingAvg)
            
        self.images=collections.OrderedDict((n,res[n]) for n in self.imagenames)
        self.update()
        
    def update(self):
        pass


def adaptImageSource(src,batchSize,inTypes,queueLength=1):
    batches=queue.Queue(queueLength)
    
    test=src.getBatch(batchSize)
    shapes=tuple(list(t.shape) for t in test)
    
    def _batchThread():
        while True:
                batches.put(src.getBatch(batchSize))
            
    batchthread=threading.Thread(target=_batchThread)
    batchthread.daemon=True
    batchthread.start()
    
    def _dequeue():
        while True:
            yield batches.get()
        
    ds = tf.data.Dataset.from_generator(_dequeue,inTypes,shapes).repeat()
    it=ds.make_one_shot_iterator()    
    return it.get_next
    

class BaseEstimator(tf.estimator.Estimator):
    def __init__(self,savedirprefix=None,runconf=None,params={}):
        self.savedir=None
        self.loss=None
        self.opt=None
        self.trainop=None
        self.net=None
        self.runconf=runconf
        self.summaries={}
        
        self.logfilename='train.log'
        self.logqueue=[]

        if savedirprefix:
            if os.path.exists(savedirprefix):
                self.savedir=savedirprefix
            else:
                self.savedir='%s-%s'%(savedirprefix,datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                
        tf.estimator.Estimator.__init__(self, model_fn=self._modelfn, model_dir=self.savedir,params=params, config=self.runconf)
        
    def log(self,*items):
        dt=datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S: ')
        msg=dt+' '.join(map(str,items))
        self.logqueue.append(msg)
        
        if os.path.isdir(self.savedir):
            with open(os.path.join(self.savedir,self.logfilename),'a') as o:
                o.write('\n'.join(self.logqueue)+'\n')
            self.logqueue=[]

    def _modelfn(self,features, labels, mode, params):

        with tf.variable_scope(tf.get_variable_scope()):
            self.createOptimizer(mode,params)
            self.createNetwork(features,labels,mode,params)
            
            if mode == tf.estimator.ModeKeys.PREDICT:
                outs={'out': tf.estimator.export.PredictOutput(self.net)}
                return tf.estimator.EstimatorSpec( mode=mode, predictions=self.net, export_outputs=outs)
            else:
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.createTrainOp(mode,params)

                self.createSummaries(mode,params)
                
                for name, image in self.summaries.items():
                    shape=[1,image.shape[0],image.shape[1],1 if len(image.shape)<3 else image.shape[2]]
                    tf.summary.image(name, tf.reshape(image, shape))
            
                self.summaries['loss']=self.loss
                tf.summary.scalar('loss',self.loss)

                return tf.estimator.EstimatorSpec(mode=mode, predictions=self.net, loss=self.loss, train_op=self.trainop, eval_metric_ops=None)
        
    def createOptimizer(self,mode,params):
        self.opt = tf.train.AdamOptimizer(params.get('learningRate',1e-3),epsilon=1e-5)
        
    def createNetwork(self,features,labels,mode,params):
        pass
    
    def createTrainOp(self,mode,params):
        pass
    
    def createSummaries(self,mode,params):
        pass
    

class BinarySegmentNN(tf.estimator.Estimator):
    def __init__(self,savedirprefix=None,runconf=None,params={}):
        self.savedir=None
        self.imgs=None
        self.masks=None
        self.loss=None
        self.logits=None
        self.preds=None
        self.opt=None
        self.trainop=None
        self.runconf=runconf
        self.summaries={}
        
        self.logfilename='train.log'
        self.logqueue=[]

        if savedirprefix:
            if os.path.exists(savedirprefix):
                self.savedir=savedirprefix
            else:
                self.savedir='%s-%s'%(savedirprefix,datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
                
        tf.estimator.Estimator.__init__(self, model_fn=self._modelfn, model_dir=self.savedir,params=params, config=self.runconf)
        
    def log(self,*items):
        dt=datetime.datetime.now().strftime('%Y%m%d-%H:%M:%S: ')
        msg=dt+' '.join(map(str,items))
        self.logqueue.append(msg)
        
        if os.path.isdir(self.savedir):
            with open(os.path.join(self.savedir,self.logfilename),'a') as o:
                o.write('\n'.join(self.logqueue)+'\n')
            self.logqueue=[]

    def _modelfn(self,features, labels, mode, params):
        global_step = tf.train.get_global_step()
        self.opt = tf.train.AdamOptimizer(params.get('learningRate',1e-3),epsilon=1e-5)
        
        try:
            self.imgs=features.values()[0]
        except:
            self.imgs=features
            
        self.masks=labels
        self.createNetwork(mode,params)
        net={'logits':self.logits,'preds':self.preds}

        with tf.variable_scope(tf.get_variable_scope(),reuse=tf.AUTO_REUSE):
            if mode == tf.estimator.ModeKeys.PREDICT:
                outs={'out': tf.estimator.export.PredictOutput(net)}
                return tf.estimator.EstimatorSpec( mode=mode, predictions=net, export_outputs=outs)
            else:
                self.loss=binaryMaskDiceLoss(self.logits,self.masks)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    self.trainop=self.opt.minimize(self.loss, global_step=global_step)

                tf.add_to_collection('endpoints',self.imgs)
                tf.add_to_collection('endpoints',self.masks)
                tf.add_to_collection('endpoints',self.logits)
                tf.add_to_collection('endpoints',self.preds)
                tf.add_to_collection('endpoints',self.trainop)
                tf.add_to_collection('endpoints',self.loss)

                self.summaries.clear()
                self.summaries['imgs'] = self.imgs[0, ..., :, :]
                self.summaries['masks'] = tf.cast(self.masks, tf.float32)[0, ..., :, :]
                self.summaries['logits'] = self.logits[0, ..., :, :,0]
                self.summaries['preds'] = tf.cast(self.preds, tf.float32)[0, ..., :, :]
                
                for name, image in self.summaries.items():
                    shape=[1,image.shape[0],image.shape[1],1 if len(image.shape)<3 else image.shape[2]]
                    tf.summary.image(name, tf.reshape(image, shape))
            
                self.summaries['loss']=self.loss
                tf.summary.scalar('loss',self.loss)

                return tf.estimator.EstimatorSpec(mode=mode, predictions=net, loss=self.loss, train_op=self.trainop, eval_metric_ops=None)
        
    def createNetwork(self,mode,params):
        pass
    
    