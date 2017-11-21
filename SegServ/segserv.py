
from __future__ import division, print_function
import sys, os, io, imageio, argparse

from flask import Flask, request, send_file

import tensorflow as tf

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from segmenter import Segmenter
from trainimagesource import rescaleArray


app = Flask(__name__)
segmap={}


@app.route('/segment/<name>', methods=['POST'])
def segment(name):
    segobj=segmap[name]
    img = request.files['image']
    
    imgmat=imageio.imread(img.stream) # read posted image file to matrix
    imgmat=rescaleArray(imgmat) # input images are expected to be normalized
    
    if imgmat.ndim==2:
        imgmat=np.expand_dims(imgmat,axis=-1)
    
    result=segobj.apply(imgmat) # apply segmentation
    
    stream=io.BytesIO()
    imageio.imwrite(stream,result,format='png') # save result to png file stream
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
    
    for name in args.metafilename:
        n=os.path.splitext(os.path.basename(name))[0]
        segmap[n]=Segmenter(name,args.device)
        
    app.run(host=args.host,port=args.port)
   