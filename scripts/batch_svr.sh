#!/bin/bash

# WARNING: out of date due to changes to wb_svr.py
# Batch execute regression.  Archives results and cleans directory
# Must be run in the directory where the data is stored

cmd="time python scripts/wb_svr.py rbf > svr_rbf.log"
echo $cmd
#eval $cmd

cmd="tar -czvf wb-svr.tar.gz rbf_dart_trial*.txt rbf_ode_trial*.txt svr_rbf.log"
echo $cmd
#eval $cmd

cmd="rm rbf_dart_trial*.txt rbf_ode_trial* svr_rbf.log"
echo $cmd
#eval $cmd

# This probably should be removed before commit
cmd="python $HOME/personal/automail.py $PWD wb-svr.tar.gz"
echo $cmd
#eval $cmd
