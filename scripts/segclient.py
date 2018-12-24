# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

'''
This is a Eidolon script used to send the selected image object to a running instance of the SegServ server and then
load the returned segmentation image. This assumes the segmentation model accepts single channel images and returns
a binary segmentation, for anything more complex the requestSeg() function should be changed.
'''
from __future__ import division, print_function
from eidolon import ImageSceneObject,processImageNp, trange, first, rescaleArray, getSceneMgr
import io

try:
    from urllib2 import Request,urlopen
except:
    from urllib.request import Request, urlopen

try:
    from imageio import imwrite, imread
except:
    from scipy.misc import imsave as imwrite,imread


# the server url, defaulting to my desktop if "--var url,<URL-to-server>" is not present on the command line
localurl=locals().get('url','http://bioeng187-pc:5000/infer/realtime')


def requestSeg(inmat,outmat,url):
    task=getSceneMgr().getCurrentTask()
    task.setMaxProgress(inmat.shape[2]*inmat.shape[3])
    task.setLabel('Segmenting...')
    count=0
    
    for s,t in trange(inmat.shape[2],inmat.shape[3]):
        count+=1
        task.setProgress(count)
            
        img=rescaleArray(inmat[:,:,s,t])
        
        if img.max()>img.min(): # non-empty image
            stream=io.BytesIO()
            imwrite(stream,img,format='png') # encode image as png
            stream.seek(0)

            request = Request(url+'?keepLargest=True',stream.read(),{'Content-Type':'image/png'})
            req=urlopen(request)
            
            if req.code==200: 
                outmat[:,:,s,t]=imread(io.BytesIO(req.read()))>0
    

if __name__=='builtins':
    o=mgr.win.getSelectedObject() or first(mgr.objs)
    
    if o is None:
        mgr.showMsg('Load and select an image object before executing this script')
    elif not isinstance(o,ImageSceneObject):
        mgr.showMsg('Selected object %r is not an image'%o.getName())
    else:
        oo=o.plugin.clone(o,o.getName()+'_Seg')
        
        with processImageNp(oo,True) as m:
            requestSeg(m,m,localurl)
                        
        mgr.addSceneObject(oo)
    