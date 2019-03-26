# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file


from __future__ import division, print_function
import subprocess, re, time, platform, threading, random
from collections import OrderedDict
from itertools import product
import inspect
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation

isWindows=platform.system().lower()=='windows'

gpunames=re.compile('\|\s+\d+\s+([a-zA-Z][^\|]+) O[nf][f ]')
gpumem=re.compile('(\d+)MiB\s*/\s*(\d+)MiB')
gpuload=re.compile('MiB\s*\|\s*(\d+)\%')


def randChoice(prob=0.5):
    '''Returns True if a randomly chosen number is less than or equal to `prob', by default this is a 50/50 chance.'''
    return random.random()<=prob


def imgBounds(img):
    '''Returns the minimum and maximum indices of non-zero lines in axis 0 of `img', followed by that for axis 1.'''
    ax0 = np.any(img, axis=0)
    ax1 = np.any(img, axis=1)
    return np.concatenate((np.where(ax0)[0][[0, -1]], np.where(ax1)[0][[0, -1]]))

    
def inBounds(x,y,margin,maxx,maxy):
    '''Returns True if (x,y) is within the rectangle (margin,margin,maxx-margin,maxy-margin).'''
    return margin<=x<(maxx-margin) and margin<=y<(maxy-margin)
    
    
def isEmpty(img):
    '''Returns True if `img' is empty, that is its maximum value is not greater than its minimum.'''
    return not (img.max()>img.min()) # use > instead of <= so that an image full of NaNs will result in True


def zeroMargins(img,margin):
    '''Returns True if the values within `margin' indices of the edges of `img' are 0.'''
    return not isEmpty(img) and not np.any(img[:,:margin]+img[:,-margin:]) and not np.any(img[:margin,:]+img[-margin:,:])


def applyArgMap(func,*posargs,**kwargs):
    '''
    Call `func' with positional arguments `posargs' and subsequent arguments replaced by named entries in `kwargs'. This
    will pull out of `kwargs' only those values keyed to the same name as an argument in `func', additional keys in 
    `argmap' are ignored if `func' does not have a ** parameter. If `func' has variable positional arguments this can 
    only be set by `posargs'.
    '''
    argspec=inspect.getfullargspec(func)
    
    if argspec.varkw is None:
        args=argspec.args
        if args[0]=='self': # if func is a constructor or method remove the self argument 
            args=args[1:]

        args=args[len(posargs):] # don't replace given positional arguments
        kwargs={k:v for k,v in kwargs.items() if k in args}
    
    return func(*posargs,**kwargs)


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


try:
    import gpustat
    def getNvidiaInfo(_=None):
        stat=gpustat.new_query()
    
        return dict(
            names=[g.name for g in stat.gpus],
            memused=[g.memory_used for g in stat.gpus],
            memtotal=[g.memory_total for g in stat.gpus],
            loads=[g.utilization for g in stat.gpus]
        )
        
except ImportError:
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


def createTestImage(width,height,numObjs=12,radMax=30,noiseMax=0.0,numSegClasses=5):
    '''
    Return a noisy 2D image with `numObj' circles and a 2D mask image. The maximum radius of the circles is given as 
    `radMax'. The mask will have `numSegClasses' number of classes for segmentations labeled sequentially from 1, plus a 
    background class represented as 0. If `noiseMax' is greater than 0 then noise will be added to the image taken from 
    the uniform distribution on range [0,noiseMax).
    '''
    image=np.zeros((width,height))
    
    for i in range(numObjs):
        x=np.random.randint(radMax,width-radMax)
        y=np.random.randint(radMax,height-radMax)
        rad=np.random.randint(5,radMax)
        spy,spx = np.ogrid[-x:width-x,-y:height-y]
        circle=(spx*spx+spy*spy)<=rad*rad
        
        if numSegClasses>1:
            image[circle]=np.ceil(np.random.random()*numSegClasses)
        else:
            image[circle]=np.random.random()*0.5+0.5
    
    norm=np.random.uniform(0,numSegClasses*noiseMax,size=image.shape)
    noisyimage=rescaleArray(np.maximum(image,norm))
    
    labels=image.astype(np.int32)
    
    return noisyimage,labels


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
    '''Rescale each array slice along the first dimension of `arr' independently.'''
    out=np.zeros(arr.shape,dtype)
    for i in range(arr.shape[0]):
        out[i]=rescaleArray(arr[i],minv,maxv,dtype)
        
    return out


def rescaleArrayIntMax(arr,dtype=np.uint16):
    '''Rescale the array `arr' to be between the minimum and maximum values of the type `dtype'.'''
    info=np.iinfo(dtype)
    return rescaleArray(arr,info.min,info.max).astype(dtype)
    
    
def cropCenter(img,*cropDims):
    '''
    Crop the center of the given array `img' to produce an array with dimensions `cropDims'. For each axis i in `img', 
    the result will have the dimension size given in cropDims[i]. If cropDims[i] is None or cropDims[i] is beyond the 
    length of `cropDims', the original dimension size is retained. Eg. cropCenter(np.zeros((10,20,20)),None,15,30)
    will return an array of dimensions (10,15,20).
    '''
    slices=[slice(None) for _ in range(img.ndim)]
    
    for i,cropdim in enumerate(cropDims):
        if cropdim is not None and cropdim>0:
            start = max(0,img.shape[i]//2-(cropdim//2)) # start of slice, 0 if img.shape[i]<cropdim 
            slices[i]=slice(start,start+cropdim)
    
    return img[tuple(slices)]


def copypasteArrays(src,dest,srccenter,destcenter,dims):
    '''
    Calculate the slices to copy a sliced area of array `src' into array `dest'. The area has dimensions `dims' (use 0 
    or None to copy everything in that dimension), the source area is centered at `srccenter' index in `src' and copied 
    into area centered at `destcenter' in `dest'. The dimensions of the copied area will be clipped to fit within the 
    source and destination arrays so a smaller area may be copied than expected. The return value is the tuples of slice 
    objects indexing the copied area in `src', and those indexing the copy area in `dest'.
    
    Example:
        src=np.random.randint(0,10,(6,6))
        dest=np.zeros_like(src)
        srcslices,destslices=copypasteArrays(src,dest,(3,2),(2,1),(3,4))
        dest[destslices]=src[srcslices]
        print(src)
        print(dest)
        
        >>> [[9 5 6 6 9 6]
             [4 3 5 6 1 2]
             [0 7 3 2 4 1]
             [3 0 0 1 5 1]
             [9 4 7 1 8 2]
             [6 6 5 8 6 7]]
            [[0 0 0 0 0 0]
             [7 3 2 4 0 0]
             [0 0 1 5 0 0]
             [4 7 1 8 0 0]
             [0 0 0 0 0 0]
             [0 0 0 0 0 0]]
    '''
    srcslices=[slice(None)]*src.ndim
    destslices=[slice(None)]*dest.ndim
    
    for i,ss,ds,sc,dc,dim in zip(range(src.ndim),src.shape,dest.shape,srccenter,destcenter,dims):
        if dim:
            d1=np.clip(dim//2,0,min(sc,dc)) # dimension before midpoint, clip to size fitting in both arrays
            d2=np.clip(dim//2+1,0,min(ss-sc,ds-dc)) # dimension after midpoint, clip to size fitting in both arrays
            
            srcslices[i]=slice(sc-d1,sc+d2)
            destslices[i]=slice(dc-d1,dc+d2)
        
    return tuple(srcslices), tuple(destslices)


def resizeCenter(img,*resizeDims,fillValue=0):
    '''
    Resize `img' by cropping or expanding the image from the center. The `resizeDims' values are the output dimensions
    (or None to use original dimension of `img'). If a dimension is smaller than that of `img' then the result will be
    cropped and if larger padded with zeros, in both cases this is done relative to the center of `img'. The result is
    a new image with the specified dimensions and values from `img' copied into its center.
    '''
    resizeDims=tuple(resizeDims[i] or img.shape[i] for i in range(len(resizeDims)))
    
    dest=np.zeros(resizeDims,img.dtype)
    dest[:]=fillValue
    srcslices,destslices=copypasteArrays(img,dest,np.asarray(img.shape)//2,np.asarray(dest.shape)//2,resizeDims)
    dest[destslices]=img[srcslices]
    
    return dest


def iterPatch(arr,patchSize,margins=(),startPos=(),copyBack=True,padMode='wrap'):
    # ensure patchSize, margins, startPos are the right length
    patchSize=(tuple(patchSize)+(0,)*arr.ndim)[:arr.ndim]
    margins=(tuple(margins)+(0,)*arr.ndim)[:arr.ndim]
    startPos=(tuple(startPos)+(0,)*arr.ndim)[:arr.ndim]
    
    # substitute sizes if None was specified in the patchSize (meaning full dimensions)
    patchSize=tuple(p or arr.shape[i] for i,p in enumerate(patchSize))
    
    # pad image by maximum values needed to ensure patches are taken from inside an image
    padWidth=tuple((p+m,)*2 for p,m in zip(patchSize,margins))
    arrpad=np.pad(arr,padWidth,padMode)
    
    ranges=tuple(range(s+m+p,d+p,p) for p,m,s,d in zip(patchSize,margins,startPos,arr.shape))
    
    for position in product(*ranges[::-1]): # reverse ranges order to iterate in index order
        slices=tuple(slice(s-m,s+p+m) for s,p,m in zip(position[::-1],patchSize,margins))
        yield arrpad[slices]
        
    if copyBack:
        slices=tuple(slice(w[0],w[0]+s) for w,s in zip(padWidth,arr.shape))
        arr[...]=arrpad[slices]
        

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
    '''
    Returns a new array with subarrays taken from `stack' in the first dimension tiled in the last two dimensions. This
    requires that `stack' have dimensions B[C][D]HW, ie. the first dimension is per-image and the last are height/width.
    '''
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


def viewVolumeJupyter(vol,figSize=(8,8),interval=250,textSize=10):
    from IPython.core.display import HTML
    
    fig, ax = plt.subplots(figsize=figSize)
    plt.axis("off")
    imgs=[]
    
    for i in range(len(vol)):
        im=plt.imshow(vol[i],animated=True)
        title=plt.text(0.5, 1.01, 'Slice %i'%i, horizontalalignment='center', 
                       verticalalignment='bottom', transform=ax.transAxes,size=textSize)
        
        imgs.append([im,title])
    
    ani=animation.ArtistAnimation(fig, imgs, interval=interval, repeat_delay=0, blit=True)
    plt.close()
    return HTML(ani.to_jshtml())


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
        
        while self.isRunning and self.is_alive() and self.step<2:
            time.sleep(0.1)
        
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

#    plt.rcParams['figure.figsize']=[6,2]
#    ax=plotSystemInfo()
    
#    src=np.random.randint(0,10,(6,6))
#    
#    print(src)
    
#    print(cropCenter(src,3,3))
#    print(cropCenter(src,10,10))
#    print(cropCenter(src,10,3))
#    
#    dest=np.zeros((10,4))
#    
#    srcslices,destslices=copypasteArrays(src,dest,np.asarray(src.shape)//2,np.asarray(dest.shape)//2,(4,5))
#    dest[destslices]=src[srcslices]
#    print(dest)
    
#    print(resizeCenter(src,10,10))
#    print(resizeCenter(src,4,4))
#    print(resizeCenter(src,4,10))
    
    im=np.zeros((64,80,3))
    
    for i,p in enumerate(iterPatch(im,(4,4))):
        print(p.shape)
        p[...]=i
        
