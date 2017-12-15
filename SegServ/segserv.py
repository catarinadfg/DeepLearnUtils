
from __future__ import division, print_function
import sys, os, io, argparse

from flask import Flask, request, send_file

import tensorflow as tf

import numpy as np

try:
    from imageio import imwrite, imread
except:
    from scipy.misc import imsave as imwrite,imread

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from segmenter import Segmenter
from trainutils import rescaleArray


app = Flask(__name__)
segmap={}


@app.route('/segment/<name>', methods=['POST'])
def segment(name):
    segobj=segmap[name]
    keepLargest=request.form.get('keepLargest','true').lower()=='true'
    img = request.files['image']
    
    imgmat=imread(img.stream) # read posted image file to matrix
    imgmat=rescaleArray(imgmat) # input images are expected to be normalized
    
    if imgmat.ndim==2: # extend a (W,H) stack to be (W,H,C) with a single channel
        imgmat=np.expand_dims(imgmat,axis=-1)
        
    result=segobj.apply(imgmat,keepLargest) # apply segmentation
    
    stream=io.BytesIO()
    imwrite(stream,result,format='png') # save result to png file stream
    stream.seek(0)
    
    return send_file(stream,'image/png') # respond with stream


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    
    parser=argparse.ArgumentParser('SegServ.py')
    parser.add_argument('metafilename',help='Path to a Tensorflow meta graph file',nargs='+')
    parser.add_argument('--host',help='Server host address',default='0.0.0.0')
    parser.add_argument('--port',help='Post to listen on',type=int,default=5000)
    parser.add_argument('--device',help='Tensorflow device name to compute on',default='/gpu:0')
    args=parser.parse_args()
    
    class EchoSegmenter(object):
        def apply(self,img,_):
            return img[...,0]
        
    segmap['echo']=EchoSegmenter() # add a "segmenter" which simply returns the first channel of any input image
    
    for name in args.metafilename:
        n=os.path.splitext(os.path.basename(name))[0]
        segmap[n]=Segmenter(name,args.device)
        
    tf.logging.info('Running server using device %r with networks %r'%(args.device,segmap.keys()))
    app.run(host=args.host,port=args.port)
    
   