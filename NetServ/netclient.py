# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

import io,json
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import numpy as np

from imageio import imwrite, imread


class InferenceClient(object):
    def __init__(self,host,port):
        self.host=str(host)
        self.port=int(port)
        self.url='%s%s:%i'%('' if '://' in self.host else 'http://',self.host,self.port)
        
        r=Request(self.url+'/list')
        req=urlopen(r)
        self.names=json.loads(req.read())
        
    def getInfo(self,name):
        '''Get the information map for the named container.'''
        r=Request('%s/info/%s'%(self.url,name))
        req=urlopen(r)
        return json.loads(req.read())
    
    def inferImage(self,name,img,**kwargs):
        '''Apply inferrence on the given image with the named container on the server.'''
        stream=io.BytesIO()
        imwrite(stream,img,format='png') # encode image as png
        stream.seek(0)

        fullurl='%s/inferimg/%s?%s'%(self.url,name,urlencode(kwargs))
        
        r=Request(fullurl,headers={'Content-Type':'image/png'},data=stream.read()) # post image data
        req=urlopen(r)
         
        return imread(io.BytesIO(req.read())) # return image read from response byte stream
    
    def inferImageVolume(self,name,vol,**kwargs):
        '''Apply inferrence on the given image volume with shape XYZT with the named container on the server.'''
        out=np.zeros_like(vol)
        
        for ind in np.ndindex(*vol.shape[2:]):
            ind=(slice(None),slice(None))+ind
            out[ind]=self.inferImage(name,vol[ind],**kwargs)
            
        return out
