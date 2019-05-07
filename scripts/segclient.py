# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

'''
This is an Eidolon script used to send the selected image object to a running instance of the SegServ server and then
load the returned segmentation image. This assumes the segmentation model accepts single channel images and returns
an integer segmentation. The `host`, `port`, and `container` values can be set using --var on the command line.

The NetServ module must be accessible for this script to work. An example usage of the script:
    
    PYTHONPATH=~/workspace/DeepLearnUtils/NetServ/ run.sh input.nii segclient.py --var container,SAX3Label
'''
import sys,os
sys.path.append(os.path.join(scriptdir,'..','NetServ'))

from eidolon import ImageSceneObject,processImageNp, trange, first, rescaleArray, getSceneMgr

import numpy as np

from netclient import InferenceClient

# local variables set using --var command line options
localhost=locals().get('host','0.0.0.0')
localport=locals().get('port','5000')
localcont=locals().get('container','echo')

client=InferenceClient(localhost,int(localport))
    

if __name__=='builtins':
    o=mgr.win.getSelectedObject() or first(mgr.objs)
    
    if o is None:
        mgr.showMsg('Load and select an image object before executing this script')
    elif not isinstance(o,ImageSceneObject):
        mgr.showMsg('Selected object %r is not an image'%o.getName())
    else:
        oo=o.plugin.clone(o,o.getName()+'_Seg')
        
        with processImageNp(oo,True) as m:
            data=rescaleArray(m,0,np.iinfo(np.uint16).max).astype(np.uint16)
            m[...]=client.inferImageVolume(localcont,data)
                        
        mgr.addSceneObject(oo)
    