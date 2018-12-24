from __future__ import division, print_function
import argparse
import nibabel
import io

try:
    from urllib2 import Request,urlopen,urlencode
except:
    from urllib.request import Request, urlopen
    from urllib.parse import urlencode

try:
    from imageio import imwrite, imread
except:
    from scipy.misc import imsave as imwrite,imread
    
import numpy as np


def requestInfer(img,url,**kwargs):
    '''
    Request inference from the server at `url' for the given image array `img' whose format must be
    suitable to encoding as a png. The `kwargs' are encoded as query arguments in the request URL.
    '''
    stream=io.BytesIO()
    imwrite(stream,img,format='png') # encode image as png
    stream.seek(0)

    request = Request(url+'?'+urlencode(kwargs),stream.read(),{'Content-Type':'image/png'})
    req=urlopen(request)

    if req.code!=200: 
        raise IOError('Error from server: %i: %s'%(req.code,req.reason))
        
    return imread(io.BytesIO(req.read()))


def inferVolume(data,url,**kwargs):
    '''Apply inference to every image of XYZT volume `data', with server URL `url' and query args `kwargs'.'''
    width,height,depth,timesteps=data.shape
    
    if data.dtype in (np.float32,np.float64):
        data=(data-data.min())/(data.max()-data.min())
        data=(data*np.iinfo(np.uint16).max).astype(np.uint16)
        
    out=np.zeros_like(data)
    
    for d in range(depth):
        for t in range(timesteps):
            out[:,:,d,t]=requestInfer(data[:,:,d,t],url,**kwargs)
            
    return out
    

if __name__=='__main__':
    parser=argparse.ArgumentParser('infernifti.py')
    parser.add_argument('infile',help='Nifti file to segment')
    parser.add_argument('outfile',help='Nifti output filename')
    parser.add_argument('--url',help='NetServ server URL',default='http://localhost:5000/infer/echo')
    args=parser.parse_args()
    
    infile=nibabel.load(args.infile)
    
    print('Applying inference from %s to %s, output to %s'%(args.url,args.infile,args.outfile))
    out=inferVolume(infile.get_data(),args.url)
    
    outfile=nibabel.Nifti1Image(out, infile.affine, infile.header)
    nibabel.save(outfile,args.outfile)
