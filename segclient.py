'''
This is a Eidolon script used to send the selected image object to a running instance of the SegServ server and then
load the returned segmentation image. This assumes the segmentation model accepts single channel images and returns
a binary segmentation, for anything more complex the requestSeg() function should be changed.
'''
from __future__ import division, print_function
from eidolon import ImageSceneObject,processImageNp, trange, first, rescaleArray
import io
import mimetools
import mimetypes
import urllib2

try:
    from imageio import imwrite, imread
except:
    from scipy.misc import imsave as imwrite,imread

mgr=mgr # pylint:disable=invalid-name,used-before-assignment

# the server url, defaulting to my desktop if "--var url,<URL-to-server>" is not present on the command line
url=locals().get('url','http://159.92.151.136:5000/segment/realtime')


def encodeMultipartFormdata(fields, files):
    '''
    Create a multipart form for POST requests needed to send files through HTTP. The `fields' dictionay maps field names
    to values to send. The `files' dictionary maps field names to (filename, data-string) pairs for files to send with 
    the form. Returns a pair (headers, body) containing the name-value header dictionary and form body string.
    '''
    boundary = mimetools.choose_boundary()
    bstr='--%s'%boundary
    lines = []

    for key, value in fields.items():
        lines += [bstr, 'Content-Disposition: form-data; name="%s"' % key, '', value]

    for key, (filename, data) in files.items():
        mimetype = mimetypes.guess_type(filename)[0] or 'application/octet-stream'
        contentdisp='Content-Disposition: form-data; name="%s"; filename="%s"' % (key, filename)
        lines += [bstr, contentdisp,'Content-Type: %s' % mimetype, '', data]

    body = '\r\n'.join(lines+['%s--'%bstr, ''])
    headers = {'Content-Type': 'multipart/form-data; boundary=%s'%boundary, 'Content-Length': str(len(body))}
    
    return headers, body


def requestSeg(inmat,outmat,url):
    task=mgr.getCurrentTask()
    task.setMaxProgress(m.shape[2]*m.shape[3])
    task.setLabel('Segmenting...')
    count=0
    
    for s,t in trange(inmat.shape[2],inmat.shape[3]):
        count+=1
        task.setProgress(count)
            
        img=rescaleArray(inmat[:,:,s,t])
        
        if img.max()>img.min(): # non-empty image
            stream=io.BytesIO()
            imwrite(stream,img*255,format='png') # encode image as png
            stream.seek(0)

            headers,body=encodeMultipartFormdata({'keepLargest':'true'},{'image':('image.png',stream.read())})
            request = urllib2.Request(url)
            request.add_data(body)
            
            for k,v in headers.items():
                request.add_header(k,v)
            
            req=urllib2.urlopen(request)
            
            if req.code==200: 
                outmat[:,:,s,t]=imread(io.BytesIO(req.read()))
    

o=mgr.win.getSelectedObject() or first(mgr.objs)

if o is None:
    mgr.showMsg('Load and select an image object before executing this script')
elif not isinstance(o,ImageSceneObject):
    mgr.showMsg('Selected object %r is not an image'%o.getName())
else:
    oo=o.plugin.clone(o,o.getName()+'_Seg')
    
    with processImageNp(oo,True) as m:
        requestSeg(m,m,url)
                    
    mgr.addSceneObject(oo)
    