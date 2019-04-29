'''
Simple script for applying inference to a Nifti file. This will produce a Nifti as output in the returned format from 
the server. The nibabel and NetServ libraries must be accessible. Example usage

    PYTHONPATH=~/workspace/nibabel:~/workspace/DeepLearnUtils/NetServ/ python infernifti.py in.nii out.nii SAX3Label
'''
from __future__ import division, print_function
import argparse
import nibabel as nib
import numpy as np

from netclient import InferenceClient
    

if __name__=='__main__':
    parser=argparse.ArgumentParser('infernifti.py')
    parser.add_argument('infile',help='Nifti file to segment')
    parser.add_argument('outfile',help='Nifti output filename')
    parser.add_argument('container',help='Hosted container name',default='echo')
    parser.add_argument('--host',help='Server host address',default='0.0.0.0')
    parser.add_argument('--port',help='Post to listen on',type=int,default=5000)
    
    args=parser.parse_args()
    
    infile=nib.load(args.infile)
    
    print('Applying inference from %s to %s, output to %s'%(args.host,args.infile,args.outfile))
    
    c=InferenceClient(args.host,args.port)
    data=infile.get_data()
    
    # rescale to the uint16 range
    if data.dtype in (np.float32,np.float64):
        data=(data-data.min())/(data.max()-data.min())
        data=(data*np.iinfo(np.uint16).max).astype(np.uint16)
        
    out=c.inferImageVolume(args.container,data)
    
    outfile=nib.Nifti1Image(out, infile.affine, infile.header)
    nib.save(outfile,args.outfile)
