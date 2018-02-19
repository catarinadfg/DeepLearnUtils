'''
This is a Eidolon script used to send the selected image object to a running instance of the SegServ server and then
load the returned segmentation image. This assumes the segmentation model accepts single channel images and returns
a binary segmentation, for anything more complex the requestSeg() function should be changed.
'''
from __future__ import division, print_function
from eidolon import ImageSceneObject,processImageNp, trange, first, rescaleArray, printFlush
import io
import urllib2

import numpy as np

try:
    from imageio import imwrite, imread
except:
    from scipy.misc import imsave as imwrite,imread

mgr=mgr # pylint:disable=invalid-name,used-before-assignment

# the server url, defaulting to my desktop if "--var url,<URL-to-server>" is not present on the command line
localurl=locals().get('url','http://bioeng187-pc:5000/segment/sharpen')


def hist_match(source, template):
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

#    source=rescaleArray(source,0,255).astype(np.uint8)
#    template=rescaleArray(template,0,255).astype(np.uint8)
#    nbr_bins=256
#    imhist,bins = np.histogram(source.flatten(),nbr_bins,normed=True)
#    tinthist,bins = np.histogram(template.flatten(),nbr_bins,normed=True)
#
#    cdfsrc = imhist.cumsum() #cumulative distribution function
#    cdfsrc = (255 * cdfsrc / cdfsrc[-1]).astype(np.uint8) #normalize
#
#    cdftint = tinthist.cumsum() #cumulative distribution function
#    cdftint = (255 * cdftint / cdftint[-1]).astype(np.uint8) #normalize
#
#    im2 = np.interp(source.flatten(),bins[:-1],cdfsrc)
#    im3 = np.interp(im2,cdftint, bins[:-1])
#
#    return rescaleArray(im3.reshape((source.shape[0],source.shape[1] )))



def requestSeg(inmat,outmat,url,patchsize=(64,64),margins=5):
    task=mgr.getCurrentTask()
    task.setMaxProgress(m.shape[2]*m.shape[3])
    task.setLabel('Segmenting...')
    count=0
    w,h=patchsize
    inmat=rescaleArray(inmat)
    mw,mh=w-margins*2,h-margins*2
    
    for s,t in trange(inmat.shape[2],inmat.shape[3]):
        count+=1
        task.setProgress(count)
            
        #img=inmat[:,:,s,t]
        img=np.zeros((inmat.shape[0]+margins*2,inmat.shape[1]+margins*2),inmat.dtype)
        img[margins:-margins,margins:-margins]=inmat[:,:,s,t]
        
        if img.max()>img.min(): # non-empty image
            iw,ih=inmat.shape[:2]
            numw=iw//mw+(1 if iw%mw!=0 else 0)
            numh=ih//mh+(1 if ih%mh!=0 else 0)
            
            for x,y in trange(numw,numh):
                subimg=img[x*mw:(x+1)*mw+margins*2,y*mh:(y+1)*mh+margins*2]
                smin=subimg.min()
                smax=subimg.max()
                
                stream=io.BytesIO()
                imwrite(stream,rescaleArray(subimg),'png') # encode image as png
                stream.seek(0)
                
                args='normalizeImg=False&keepLargest=False&resultScale=1.0'
                request = urllib2.Request(url+'?'+args,stream.read(),{'Content-Type':'image/png'})
                req=urllib2.urlopen(request)
                
                if req.code==200: 
                    im=imread(io.BytesIO(req.read()))
                    printFlush(im.shape,im.min(),im.max(),np.average(im),smin,smax)
                    #im=np.maximum(im.astype(float)-im[0,0],0)
                    #outmat[x*w:(x+1)*w,y*h:(y+1)*h,s,t]=rescaleArray(im,smin,smax)
                    outmat[x*mw:(x+1)*mw,y*mh:(y+1)*mh,s,t]=hist_match(rescaleArray(im[margins:-margins,margins:-margins]),subimg[margins:-margins,margins:-margins])
                    
                
#            stream=io.BytesIO()
#            imwrite(stream,img,'png') # encode image as png
#            stream.seek(0)
#
#            request = urllib2.Request(url,stream.read(),{'Content-Type':'image/png'})
#            req=urllib2.urlopen(request)
#            
#            if req.code==200: 
#                im=imread(io.BytesIO(req.read()))
#                printFlush(im.shape,im.min(),im.max(),im[0,0])
#                im=np.maximum(im.astype(float)-im[0,0],0)
#                outmat[:,:,s,t]=rescaleArray(im)
            

o=mgr.win.getSelectedObject() or first(mgr.objs)

if o is None:
    mgr.showMsg('Load and select an image object before executing this script')
elif not isinstance(o,ImageSceneObject):
    mgr.showMsg('Selected object %r is not an image'%o.getName())
else:
    oo=o.plugin.clone(o,o.getName()+'_sharp')
    
    with processImageNp(oo,True) as m:
        requestSeg(m,m,localurl)
                    
    mgr.addSceneObject(oo)
    