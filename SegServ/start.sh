# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

#! /bin/bash

# these are paths to meta graph files that I have for my networks, replace these with your own to start the server
#REALTIME=/home/localek10/data/Projs/BramRealTime/realtime-20180125202714/model.ckpt-3001.meta
#REALTIME=/home/localek10/data/Projs/BramRealTime/realtime-20180205030522/model.ckpt-4001.meta
#REALTIME=/home/localek10/data/Projs/BramRealTime/realtime-20180223180919/model.ckpt-8001.meta

# these are experimental models of mine
#DLTK=/home/localek10/workspace/Projs/SegmentTests/dltk-20171114184156/dltk.meta
#SAX=/home/localek10/data/Projs/SAXSegment/saxown-20180202005555/model.ckpt-2401.meta
#LAX=/home/localek10/data/Projs/LAXSegment/lax-20180125133642/model.ckpt-3001.meta
#SHARPEN=/data/Projs/BramRealTime/sharpen-20180219183135/model.ckpt-10103.meta

# for the realtime reconstruction workflow
REALTIME=/data/Projs/BramRealTime/realtime-20180226141303-24003/model.ckpt-24003.meta


# restricts the server to using only device 1, this is more effective than using tf.device() it seems, comment if you don't want to use device 1
export CUDA_VISIBLE_DEVICES=1

kill $(cat pid.log) || echo "Server not running, nothing to kill"

nohup python segserv.py realtime:$REALTIME &

echo $! > pid.log

tail -f nohup.out
