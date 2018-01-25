#! /bin/bash

# these are paths to meta graph files that I have for my networks, replace these with your own to start the server
REALTIME=/home/localek10/workspace/Projs/BramRealTime/realtime-20171108162848/realtime.meta
DLTK=/home/localek10/workspace/Projs/SegmentTests/dltk-20171114184156/dltk.meta
SAX=/home/localek10/data/Projs/SAXSegment/sax-20180119145547/model.ckpt-14228.meta

# restricts the server to using only device 1, this is more effective than using tf.device() it seems, comment if you don't want to use device 1
export CUDA_VISIBLE_DEVICES=1

nohup $HOME/anaconda2/bin/python segserv.py realtime:$REALTIME sax:$SAX 2> segserv.log &

echo $! > pid.log
