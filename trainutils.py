
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


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
    

def plotGraphImages(graphtitle,graphmap,imagemap,yscale='log',fig=None):
    '''
    Plot graph data with images below in a pyplot figure. 
    '''
    numimages=len(imagemap)
    assert numimages>0
    
    if fig:
        fig.clf()
    else:
        fig = plt.figure(figsize=(20,14))
    
    gs = gridspec.GridSpec(2,numimages,height_ratios=[2, 3],wspace=0.02, hspace=0.05)
    
    graph=plt.subplot(gs[0,:])
    
    for n,v in graphmap.items():
        graph.plot(v,label='%s = %.3f '%(n,v[-1]))

    graph.set_title(graphtitle)
    graph.set_yscale(yscale)
    graph.axis('on')
    graph.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    graph.grid(True,'both')
    graph.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ims=[]
    for i,n in enumerate(imagemap):
        im=plt.subplot(gs[1,i])
        im.imshow(np.squeeze(imagemap[n]),cmap='gray')
        im.set_title('%s = %.3f -> %.3f'%(n,imagemap[n].min(),imagemap[n].max()))
        im.axis('off')
        ims.append(im)
        
    return fig,[graph]+ims

    
if __name__=='__main__':
    im1=np.random.rand(5,10)
    im2=np.random.rand(15,15)
    plotGraphImages('graph',{'x':[0,1,2,1],'y':[4,5,0,-1]},{'im1':im1,'im2':im2})
    