# NetServ

This simple program is for serving neural network models over HTTP using Flask. This allows clients to submit data to be 
segmented or otherwise processed and get results back. The command line is used to setup the models to serve by specifying 
initialization script files. 

For example, the following would start the server with a script file which implements the setup protocol:

    python segserv.py init.py
    
The included shell script is used to start the server then records the PID to a file, which is used to kill a running
server when the script is run again:

    ./start.sh init.py
    
See `python netserv.py --help` for other command line options.