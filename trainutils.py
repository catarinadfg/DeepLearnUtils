
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def rescaleArray(arr,minv=0.0,maxv=1.0):
    '''Rescale the values of numpy array `arr' to the range `minv' to `maxv'.'''
    mina=np.min(arr)
    norm=(arr-mina)/(np.max(arr)-mina)
    return (norm*(maxv-minv))+minv


def comparePrediction(imgs,masks,logits,preds,title=''):
    figax=[]
    for index in range(preds.shape[0]):
        im=imgs[index,...,0]
        logit=np.squeeze(logits[index])
        pred=np.squeeze(preds[index])
                
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
        
        if masks is not None:
            ax[0].imshow(im+masks[index]*0.5)
            ax[3].imshow(np.stack([masks[index],pred*masks[index],pred],axis=2))
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
    

def plotGraphImages(graphtitle,graphmap,imagemap,yscale='log'):
    numimages=len(imagemap)
    
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(2,numimages,height_ratios=[1, 3],wspace=0.1, hspace=0.05)
    
    graph=plt.subplot(gs[0,:])
    graph.set_title(graphtitle)
    graph.set_yscale(yscale)
    graph.axis('on')
    
    for n,v in graphmap.items():
        graph.plot(v,label=n)
    
    ims=[]
    for i,n in enumerate(sorted(imagemap)):
        im=plt.subplot(gs[1,i])
        im.imshow(np.squeeze(imagemap[n]),cmap='gray')
        im.set_title(n)
        im.axis('off')
        ims.append(im)
        
    return fig,[graph]+ims

    
if __name__=='__main__':
    im1=np.random.rand(5,10)
    im2=np.random.rand(15,15)
    plotGraphImages('graph',{'x':[0,1,2,1],'y':[4,5,0,-1]},{'im1':im1,'im2':im2})
    