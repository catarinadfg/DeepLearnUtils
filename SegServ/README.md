# SegServ

This simple program is for serving (primarily) segmentation models over HTTP using Tensorflow and Flask. This allows clients to submit data to be segmented and get results back. The command line is used to setup the models to serve by specifying names for model files. For example, the following would start the server with a single model called seg loaded from seg.meta:

    python segserv.py seg:seg.meta
    
See `python segserv.py --help` for other command line options.