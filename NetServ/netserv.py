# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

'''
A simple Flask-based server for providing neural network inference for images through a HTTP interface. The routes
/infer/<name> represent networks for applying inference on single images at a time to produce output images. An image
in PNG format is POSTed to this address and a PNG is sent back as the response. Other arguments can be passed in the URL.
The actual operation of the network is defined by callables accepting the input image (2D Numpy array) as its first 
argument followed by expanded keyword arguments, and returning the resulting image as a 2D array. Input values are those
stored in the input PNG file and output values should be suitable for storing into the output PNG.

The server is initialized with such inference objects by importing given script files as modules and calling the defined
function getInferObjects(). This function should return a dictionary relating network names to their inference objects to
be added to the server. Example script producing a "network" which returns the input image:
    
def echo(img,**kwargs):
    return img
    
def getInferObjects():
    return {"echo" : echo}
    
This echo object would be accessed through URL path /infer/echo. 

A running server can be tested with an input image "input.png" as such:
    
    curl -X POST --data-binary "@input.png" -H "Content-Type:image/png" localhost:5000/infer/echo -o output.png
    
or using forms as such:
    
    curl -F "data=@input.png" localhost:5000/infer/echo -o output.png

'''
from __future__ import division, print_function
import io, argparse, ast, logging, importlib.util

from flask import Flask, request, send_file

import numpy as np

from imageio import imwrite, imread

app = Flask(__name__)

infermap={
        'echo':lambda i,**kw:np.squeeze(i) # add a "network" which simply returns the first channel of any input image
}


@app.route('/infer/<name>', methods=['POST'])
def infer(name):
    obj=infermap[name]
    args={k:ast.literal_eval(v) for k,v in request.args.items()} # keep only one value per argument name
        
    data=request.data or request.files['data'].read()
    imgmat=imread(io.BytesIO(data)) # read posted image file to matrix
    logging.info('infer(): %r %r %r %r %r %r'%(name,imgmat.shape,imgmat.dtype,imgmat.min(),imgmat.max(),args))

    result=obj(imgmat,**args) # apply inference
    
    stream=io.BytesIO()
    imwrite(stream,result,format='png') # save result to png file stream
    stream.seek(0)
    
    return send_file(stream,'image/png') # respond with stream


if __name__=='__main__':
    parser=argparse.ArgumentParser('netserv.py',description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('scripts',help='Script files to import as modules for initialization',nargs='*')
    parser.add_argument('--host',help='Server host address',default='0.0.0.0')
    parser.add_argument('--port',help='Post to listen on',type=int,default=5000)
    args=parser.parse_args()
    
    for i,script in enumerate(args.scripts):
        # load script as module
        spec=importlib.util.spec_from_file_location("initmod%i"%i, script)
        mod=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        
        # update infermap with the returned inference objects
        infermap.update(mod.getInferObjects())
        
    logging.info('Running server with networks %r'%(list(infermap.keys()),))
    app.run(host=args.host,port=args.port)
    
   