# DeepLearnUtils 
# Copyright (c) 2017-8 Eric Kerfoot, KCL, see LICENSE file

#! /bin/bash

dir=$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )

kill $(cat pid.log) || echo "Server not running, nothing to kill"

nohup python $dir/netserv.py $* &

echo $! > pid.log

tail -f nohup.out
