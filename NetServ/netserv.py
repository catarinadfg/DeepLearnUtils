# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

'''
A simple Flask-based server for providing neural network inference for images through a HTTP interface. The routes
/inferimg/<name> represent networks for applying inference on single images at a time to produce output images. An image
in PNG format is POSTed to this address and a PNG is sent back as the response. Other arguments can be passed in the URL.
The actual operation of the network is defined by instances of InferenceContainer accepting the input image (2D/3D Numpy 
array) through their infer() method followed by expanded keyword arguments, and returning the resulting image as a 2D/3D 
array. Input values are those stored in the input PNG file and output values should be suitable for storing into the 
output PNG.

The server is initialized with such inference objects by importing given script files as modules and calling the defined
function getContainers(). This function should return a list of container objects with unique names to be added to the 
server. Example script producing a "network" which returns the input image:
    
    
class EchoContainer(InferenceContainer):
    def __init__(self):
        super().__init__('echo','Echo test container',{'in':(0,0,3)},{'out':(0,0,3)},{})
        
    def infer(self,*inputMatrices,**kwargs):
        return np.squeeze(inputMatrices[0])
    
def getContainerMap():
    return [EchoContainer()]
    
    
This echo object would be accessed through URL path /inferpng/echo. 

A running server can be tested with an input image "input.png" as such:
    
    curl -X POST --data-binary "@input.png" -H "Content-Type:image/png" localhost:5000/inferimg/echo -o output.png
    
or using forms as such:
    
    curl -F "data=@input.png" localhost:5000/inferimg/echo -o output.png

'''
from __future__ import division, print_function
import io, argparse, ast, logging, importlib.util

from flask import Flask, request, send_file, jsonify

import numpy as np

from imageio import imwrite, imread


class InferenceContainer(object):
    def __init__(self,name,description,inputMap,outputMap,argMap):
        self.name=name
        self.description=description
        self.inputMap=inputMap
        self.outputMap=outputMap
        self.argMap=argMap
        
    def infer(self,*inputMatrices,**kwargs):
        pass
    
    
class EchoContainer(InferenceContainer):
    def __init__(self):
        super().__init__('echo','Echo test container',{'in':(0,0,3)},{'out':(0,0,3)},{})
        
    def infer(self,*inputMatrices,**kwargs):
        return np.squeeze(inputMatrices[0])
    

app = Flask(__name__)
containers={ 'echo': EchoContainer() }


@app.route('/list')
def listContainers():
    '''Returns a list of the loaded container names.'''
    return jsonify(list(containers.keys()))
    

@app.route('/info/<name>')
def info(name):
    '''Returns the information map for the named container.'''
    obj=containers[name]
    
    infoMap={
        'name':obj.name,
        'description':obj.description,
        'inputs':obj.inputMap,
        'outputs':obj.outputMap,
        'arguments':obj.argMap
    }
    
    return jsonify(infoMap)


@app.route('/inferimg/<name>', methods=['POST'])
def inferimg(name):
    obj=containers[name]
    args={k:ast.literal_eval(v) for k,v in request.args.items()} # keep only one value per argument name
        
    data=request.data or request.files['in'].read() # read posted data or a form file called 'data'
    imgmat=imread(io.BytesIO(data)) # read posted image file to matrix
    logging.info('infer(): %r %r %r %r %r %r'%(name,imgmat.shape,imgmat.dtype,imgmat.min(),imgmat.max(),args))

    result=obj.infer(imgmat,**args) # apply inference
    
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
        
        # update containers with the returned inference objects
        containers.update({c.name:c for c in mod.getContainers()})
        
    logging.info('Running server with networks %r'%(list(containers.keys()),))
    app.run(host=args.host,port=args.port)
    
   