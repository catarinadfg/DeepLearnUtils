
import numpy as np
import matplotlib.pyplot as plt


def rescaleArray(arr,minv=0.0,maxv=1.0):
    '''Rescale the values of numpy array `arr' to the range `minv' to `maxv'.'''
    mina=np.min(arr)
    norm=(arr-mina)/(np.max(arr)-mina)
    return (norm*(maxv-minv))+minv


def comparePrediction(imgs,masks,logits,preds,title=''):
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

        
def viewImages(img,mask,maskval=0.25):
    import pyqtgraph as pg
    
    if mask is not None:
        mask=mask*(maskval*img.max())
        if img.ndim==3 and img.shape[2]>1:
            mask=np.stack([mask]*img.shape[2],axis=-1)
            
        img=img+mask
        
    pg.image(img,xvals=np.arange(img.shape[0]+1))
    