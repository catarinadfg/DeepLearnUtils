# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file


from __future__ import division, print_function
import subprocess, re, time, platform, threading
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

isWindows=platform.system().lower()=='windows'

gpunames=re.compile('\|\s+\d+\s+([a-zA-Z][^\|]+) O[nf][f ]')
gpumem=re.compile('(\d+)MiB\s*/\s*(\d+)MiB')
gpuload=re.compile('MiB\s*\|\s*(\d+)\%')


def oneHot(labels,numClasses):
    '''Converts label image `labels' to a one-hot vector with `numClasses' number of channels as last dimension.'''
    labels=labels%numClasses
    y = np.eye(numClasses)
    onehot=y[labels.flatten()]
    
    return onehot.reshape(tuple(labels.shape)+(numClasses,))


def iouMetric(a,b,smooth=1e-5):
    '''
    Returns the intersection-over-union metric between `a' and `b', which is 0 if identical, 1 if `a' is completely
    disjoint from `b', and values in between if there is overlap. Both inputs are thresholded to binary masks.
    '''
    a=a==a.max()
    b=b==b.max()
    
    inter=a*b
    union=a+b
    
    return 1.0-(inter.sum()+smooth)/(union.sum()+smooth)


def getNvidiaInfo(proc='nvidia-smi'):
    '''
    Get name, nemory usage, and loads on GPUs using the program "nvidia-smi". The value `proc' should be the path to the
    program, which must be absolute if it isn't on the path. The return value is a dictionary with a list of GPU names
    keyed to "names", a list of per-GPU memory use values in MiB keyed to "memused", a list of total memory keyed to 
    "memtotal", and a list of percentage compute loads keyed to "loads".
    '''
    result=str(subprocess.check_output(proc))
    
    names=re.findall(gpunames,result)
    load=re.findall(gpuload,result)
    mem=re.findall(gpumem,result)
    
    return dict(
        names=[m.strip() for m in names],
        memused=[int(m[0]) for m in mem],
        memtotal=[int(m[1]) for m in mem],
        loads=[int(l) for l in load]
    )    
    

def getMemInfo(src='/proc/meminfo'):
    '''Return a dictionary containing the parsed contents of `src', which is expected to be like /proc/meminfo.'''
    def split(v):
        v=v.split()
        return v[0][:-1],int(v[1])
    
    with open(src) as o:
        return dict(map(split,o))


def getCpuInfo(src='/proc/stat',waitTime=0.05):
    '''Use info in `src' to generate a dictionary mapping CPUs to (load%,idle time, total time) tuples.'''
    def getCpuTimes():
        with open(src) as o:
            result={}
            
            for line in o:
                if not line.startswith('cpu'):
                    break
                
                line=[l.strip() for l in line.split(' ') if l]
                values=[int(v) for v in line[1:]] # user,nice,system,idle,iowait,irq,softrig,steal,guest,guest_nice
                
                result[line[0]]=(values[3]+values[4],sum(values[:-2])) # (idle times, total times without guest values)
                
            return result
            
    start=getCpuTimes()
    time.sleep(waitTime)
    stop=getCpuTimes()
    result = {}

    for cpu in start:
        curIdle,curTotal = stop[cpu]
        prevIdle,prevTotal = start[cpu]

        load=((curTotal-prevTotal)-(curIdle-prevIdle))/float(curTotal-prevTotal)*100
        result[cpu]=(load,curIdle,curTotal)
        
    return result


def rescaleArray(arr,minv=0.0,maxv=1.0,dtype=np.float32):
    '''Rescale the values of numpy array `arr' to be from `minv' to `maxv'.'''
    if dtype is not None:
        arr=arr.astype(dtype)
        
    mina=np.min(arr)
    maxa=np.max(arr)
    
    if mina==maxa:
        return arr*minv
    
    norm=(arr-mina)/(maxa-mina) # normalize the array first
    return (norm*(maxv-minv))+minv # rescale by minv and maxv, which is the normalized array by default


def rescaleInstanceArray(arr,minv=0.0,maxv=1.0,dtype=np.float32):
    out=np.zeros_like(arr)
    for i in range(arr.shape[0]):
        out[0]=rescaleArray(arr[0],minv,maxv,dtype)
        
    return out


def rescaleArrayIntMax(arr,dtype=np.uint16):
    '''Rescale the array `arr' to be between the minimum and maximum values of the type `dtype'.'''
    info=np.iinfo(dtype)
    return rescaleArray(arr,info.min,info.max).astype(dtype)


def cropCenter(img,cropy,cropx,startAxis=0):
    '''
    Crop the center of the given array `img' along the dimensions `startAxis' and `startAxis'+1.
    The result will have axis `startAxis' dimension `cropy' and axis `startAxis'+1 dimension `cropx'.
    '''
    y,x = img.shape[1:3]
    starty = max(0,y//2-(cropy//2))
    startx = max(0,x//2-(cropx//2))
    
    slices=[slice(None) for _ in range(img.ndim)]
    slices[startAxis]=slice(starty,starty+cropy)
    slices[startAxis+1]=slice(startx,startx+cropx)
    
    return img[slices]
    #return img[starty:starty+cropy,startx:startx+cropx]
    
    
def cropCenterAxes(img,*cropDims):
    '''
    Crop the center of the given array `img' to produce an array with dimensions `cropDims'. For each axis i in `img', 
    the result will have the dimension size given in cropDims[i]. If cropDims[i] is None or cropDims[i] is beyond the 
    length of `cropDims', the original dimension size is retained. Eg. cropCenterAxes(np.zeros((10,20,20)),None,15,30)
    will return an array of dimensions (10,15,20).
    '''
    slices=[slice(None) for _ in range(img.ndim)]
    
    for i,cropdim in enumerate(cropDims):
        if cropdim is not None and cropdim>0:
            start = max(0,img.shape[i]//2-(cropdim//2)) # start of slice, 0 if img.shape[i]<cropdim 
            slices[i]=slice(start,start+cropdim)
    
    return img[tuple(slices)]    


def flatten4DVolume(im):
    '''Given a volume in HWDT ordering, reshape dimensions D and T to a single D dimension and reorder result axes to DHW.'''
    return im.reshape((im.shape[0],im.shape[1],-1)).transpose((2,0,1))


def stackImages(images,cropy,cropx,dtype=np.float32):
    '''
    Create a 3D array from the 3D arrays in `images' by cropping each to (D,cropy,cropx) in shape (for depth axis 
    dimension D) and then stacking them together. This assumes that each array in `images' is in DHW axis ordering.
    '''
    totalsize=sum(i.shape[0] for i in images)
    output=np.zeros((totalsize,cropy,cropx),dtype)
    pos=0
    
    for im in images:
        im=cropCenter(im,cropy,cropx,1)
        d,y,x=im.shape
        
        cout=output[pos:pos+d]
        cout=cropCenter(cout,y,x,1)
        cout[:]=im
        pos+=d
        
    return output


def tileStack(stack,cols,rows):
    b=stack.shape[0]
    stack=stack[:b-b%cols]
    rows=min(rows,stack.shape[0]//cols)
    out=[]
    
    for r in range(rows):
        out.append([stack[r*cols+c] for c in range(cols)])
        
    return np.block(out)


def comparePrediction(imgs,masks,logits,preds,title=''):
    '''
    Expects `imgs' of shape BWHC, all others of shape BWH
    '''
    figax=[]
    for index in range(preds.shape[0]):
        im=imgs[index,...,0]
        mask=masks is not None and masks[index]
        logit=np.squeeze(logits[index])
        pred=np.squeeze(preds[index])
        
        assert im.ndim==logit.ndim
        assert logit.ndim==pred.ndim
        assert mask is None or mask.ndim==im.ndim
        
        padx=int(im.shape[0]-pred.shape[0])//2
        pady=int(im.shape[1]-pred.shape[1])//2
        if padx and pady:
            pred=np.pad(pred,((padx,padx),(pady,pady)),'constant',constant_values=np.min(pred))
            
        fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20,6))
        ax[0].axis('off')
        ax[1].axis('off')
        ax[2].axis('off')
        ax[3].axis('off')
        ax[0].set_title(title)
        
        ax[1].imshow(logit)
        ax[2].imshow(pred)
        
        if mask is not None:
            st=np.stack([masks[index],pred*masks[index],pred],axis=2).astype(np.float32)
            
            ax[0].imshow(im+mask*0.5)
            ax[3].imshow(st)
        else:
            ax[0].imshow(im)
            ax[3].imshow(im+pred*0.5)
            
        figax.append((fig,ax))
        
    return figax

        
def viewImages(img,mask,maskval=0.25):
    import pyqtgraph as pg
    
    if mask is not None:
        mask=mask*(maskval*img.max())
        if img.ndim==3 and img.shape[2]>1:
            mask=np.stack([mask]*img.shape[2],axis=-1)
            
        img=img+mask
        
    p=pg.image(img,xvals=np.arange(img.shape[0]+1))
    p.parent().showMaximized()
    
    def _keypress(e):
        if e.key()==pg.QtCore.Qt.Key_Escape:
            p.parent().close()
            
    setattr(p.parent(),'keyPressEvent',_keypress)
    return p


def plotSystemInfo(ax=None):
    ax=ax or plt.subplot()
    cols=[]
    labels=['CPU Load','Mem Alloc']
    colors=['r','b']
    
    try:
        cpu=getCpuInfo()
        mem=getMemInfo()
    
        allocperc=int((1.0-float(mem['MemAvailable'])/mem['MemTotal'])*100)
        
        cols+=[cpu['cpu'][0], allocperc]
    except:
        cols+=[0,0]
    
    try:
        gpu=getNvidiaInfo()
        for n in range(len(gpu['names'])):
            load=gpu['loads'][n]
            name=gpu['names'][n]
            memper=int(100*(float(gpu['memused'][n])/gpu['memtotal'][n]))
            
            cols+=[load,memper]
            labels+=[name+' Load',name+' Mem']
            colors+=['r','b']
    except:
        pass
        
    inds=np.arange(len(cols))
    
    bars=ax.barh(inds,cols,height=0.3)
    
    for b,c in zip(bars,colors):
        b.set_facecolor(c)
    
    ax.invert_yaxis()
    ax.set_yticks(inds)
    ax.set_yticklabels(labels)
    ax.set_xlim([0, 100])
    ax.grid(True,axis='x')
    ax.set_xlabel('% Usages')
    ax.set_title('System Info')
    
    return ax
    

def plotGraphImages(graphtitle,graphmap,imagemap,yscale='log',fig=None):
    '''
    Plot graph data with images below in a pyplot figure. 
    '''
    gridshape=(4, max(1,len(imagemap)))
    
    if fig is not None:
        fig.clf()
    else:
        fig = plt.figure(figsize=(20,10))
    
    graph= plt.subplot2grid(gridshape, (0, 0),colspan=gridshape[1])
   
    for n,v in graphmap.items():
        graph.plot(v,label='%s = %.5g '%(n,v[-1]))

    graph.set_title(graphtitle)
    graph.set_yscale(yscale)
    graph.axis('on')
    graph.legend(bbox_to_anchor=(1, 1), loc=1, borderaxespad=0.)
    graph.grid(True,'both')
    graph.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ims=[graph]
    for i,n in enumerate(imagemap):
        im=plt.subplot2grid(gridshape, (1, i),rowspan=2)
        
        im.imshow(np.squeeze(imagemap[n]),cmap='gray')
        im.set_title('%s\n%.3g -> %.3g'%(n,imagemap[n].min(),imagemap[n].max()))
        im.axis('off')
        ims.append(im)
    
    ax=plt.subplot2grid(gridshape, (3, 0),rowspan=1,colspan=gridshape[1])
    
    if not isWindows:
        ax=plotSystemInfo(ax)
        ims.append(ax)
    
    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    
    return fig,ims
    

class JupyterThreadMonitor(threading.Thread):
    def __init__(self,*args,**kwargs):
        threading.Thread.__init__(self,*args,**kwargs)
        
        from IPython.core.display import display, clear_output
        self._display=display
        self._clear_output=clear_output
        self.graphVals=OrderedDict()
        self.imageVals=OrderedDict()
        self.graphAvgLen=50
        self.step=0
        self.isRunning=True
        self.fig=None
        self.lock=threading.Lock()
        
    def stop(self):
        self.isRunning=False
        self.join()
        
    def updateGraphVals(self,vals,calcAvgs=()):
        with self.lock:
            for k,v in vals.items():
                self.graphVals.setdefault(k,[]).append(v)
                
                if k in calcAvgs:
                    av=np.average(self.graphVals[k][-self.graphAvgLen:])
                    self.graphVals.setdefault(k+' (Avg)',[]).append(av)
                
    def updateImageVals(self,vals):
        with self.lock:
            self.imageVals.update(vals)
            
    def displayMonitor(self,delay=10.0,doOnce=False):
        self.fig=None
        title='Train Values Step '
        
        while self.isRunning and self.is_alive() and not doOnce:
            with self.lock:
                self.fig,ax=plotGraphImages('%s%i'%(title,self.step,),self.graphVals,self.imageVals,fig=self.fig)
                
            self._clear_output(wait=True)
            self._display(plt.gcf())
            time.sleep(delay)
            
        with self.lock:
            suffix='' if self.is_alive() else' (Stopped)'
            self.fig,ax=plotGraphImages('%s%i%s'%(title,self.step,suffix),self.graphVals,self.imageVals,fig=self.fig)
            
        self._display(plt.gcf())
        self._clear_output(wait=True)
        
    def status(self):
        graphvals={k:v[-1] for k,v in self.graphVals.items()}
        msg='Alive: %s, Step: %i, Values: %r'%(self.is_alive(),self.step,graphvals)
        return msg
            
    
if __name__=='__main__':
#    im1=np.random.rand(5,10)
#    im2=np.random.rand(15,15)
#    fig=plt.figure(figsize=(8,6))
#    
#    f,ims=plotGraphImages('graph',{'x':[0,1,2,1],'y':[4,5,0,-1]},{'im1':im1},fig=fig)
#    plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
    
#    print(getNvidiaInfo())
#    print(getMemInfo())
#    print(getCpuInfo())
#    
#    for cpu,load in getCpuInfo().items():
#        print(cpu,load)

    plt.rcParams['figure.figsize']=[6,2]
    ax=plotSystemInfo()
    