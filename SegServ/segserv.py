'''
A simple Flask-based server for providing segmentation inference through a HTTP interface. 
'''
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


app = Flask(__name__)
segmap={}


@app.route('/segment/<name>', methods=['POST'])
def segment(name):
    segobj=segmap[name]
    args=dict(request.args.items()) # keep only one value per argument name
    
    imgmat=imread(request.data) # read posted image file to matrix
        
    tf.logging.info('segment(): %r %r %r %r %r %r'%(name,imgmat.shape,imgmat.dtype,imgmat.min(),imgmat.max(),args))
    result=segobj.apply(imgmat,**args) # apply segmentation
    
    stream=io.BytesIO()
    imwrite(stream,result,format='png') # save result to png file stream
    stream.seek(0)
    
    return send_file(stream,'image/png') # respond with stream


if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    
    parser=argparse.ArgumentParser('SegServ.py')
    parser.add_argument('metafilename',help='Named path to a Tensorflow meta graph file in the form "name:path"',nargs='*')
    parser.add_argument('--host',help='Server host address',default='0.0.0.0')
    parser.add_argument('--port',help='Post to listen on',type=int,default=5000)
    parser.add_argument('--device',help='Tensorflow device name to compute on',default='/gpu:0')
    args=parser.parse_args()
    
    class EchoSegmenter(object):
        def apply(self,img):
            return np.squeeze(img)
        
    segmap['echo']=EchoSegmenter() # add a "segmenter" which simply returns the first channel of any input image
    
    for namepath in args.metafilename:
        name,path=namepath.split(':')
        segmap[name]=Segmenter(path,args.device)
        
    tf.logging.info('Running server using device %r with networks %r'%(args.device,segmap.keys()))
    app.run(host=args.host,port=args.port)
    
   