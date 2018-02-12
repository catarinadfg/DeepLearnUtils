'''
This is a Eidolon script used to send the selected image object to a running instance of the SegServ server and then
load the returned segmentation image. 

This script is specifically for a network trained to accept a 3 channel image where the channels are the greyscale image
data, FFT motion information, and edge filter image. These inputs are generated from a time-dependent image object by the 
function processImage() below to create a single frame input stack. The original training images were generated in the 
same way to produce single-frame input stacks from time-dependent stacks. The requestSeg() function is also different
from that in segclient.py to account for this.
'''
from __future__ import division, print_function
from eidolon import ImageSceneObject, processImageNp, first, rescaleArray, calculateMotionField
import io
import urllib2

import numpy as np
from scipy import ndimage

try:
    from imageio import imwrite, imread
except:
    from scipy.misc import imsave as imwrite,imread

mgr=mgr # pylint:disable=invalid-name,used-before-assignment

# the server url, defaulting to my desktop if "--var url,<URL-to-server>" is not present on the command line
localurl=locals().get('url','http://bioeng187-pc:5000/segment/sax')


def processImage(obj):
    with processImageNp(obj,False) as imat:
        mag=rescaleArray(imat[...,0])
        motion=rescaleArray(calculateMotionField(obj))
        edge=rescaleArray(ndimage.generic_gradient_magnitude(mag,ndimage.sobel))
        
        return mag,motion,edge
    
    
def requestSeg(inmat,outmat,url):
    task=mgr.getCurrentTask()
    task.setMaxProgress(inmat.shape[2])
    task.setLabel('Segmenting...')
    
    count=0
    for s in range(inmat.shape[2]):
        count+=1
        task.setProgress(count)
            
        img=inmat[:,:,s,:]
        
        if img.max()>img.min(): # non-empty image
            stream=io.BytesIO()
            imwrite(stream,img*255,format='png') # encode image as png
            stream.seek(0)

            request = urllib2.Request(url+'?keepLargest=true',stream.read(),{'Content-Type':'image/png'})
            req=urllib2.urlopen(request)
            
            if req.code==200: 
                outmat[:,:,s]=imread(io.BytesIO(req.read()))
    

o=mgr.win.getSelectedObject() or first(mgr.objs)

if o is None:
    mgr.showMsg('Load and select an image object before executing this script')
elif not isinstance(o,ImageSceneObject):
    mgr.showMsg('Selected object %r is not an image'%o.getName())
else:
    oo=o.plugin.extractTimesteps(o,o.getName()+'_Seg',indices=[0])
    
    mag,motion,edge=processImage(o)
    
    combined=np.stack([mag,motion,edge],axis=-1)
    
    #minx,miny,maxx,maxy=calculateBinaryMaskBox(mag)
    inds=ndimage.find_objects(mag.astype(int))[0]
    minx,miny,maxx,maxy=inds[0].start, inds[1].start, inds[0].stop, inds[1].stop
    
    combined[miny:maxy,minx:maxx]=0
    
    with processImageNp(oo,True) as m:
        requestSeg(combined,m[...,0],localurl)
                    
    mgr.addSceneObject(oo)
    