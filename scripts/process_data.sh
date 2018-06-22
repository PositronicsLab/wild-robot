#!/bin/bash

scriptpath=$1

#echo $scriptpath

for tarfile in *.tar.gz; do
  fields=(${tarfile//./ })
  root=${fields[0]} 
  params=(${root//_/ })
  #sim=${params[0]}
  #s=${params[1]}
  #offset=(${s//offset/ })
  #echo $sim
  #echo $offset

  #untar
  echo "Untarring $tarfile"
  cmd="tar -xvzf $tarfile"
  echo $cmd
  eval $cmd

  #decimate state
  echo "Decimating sim.log to reduce size into sim.100fps"
  cmd="python ${scriptpath}decimate_state.py 999 1000 sim.log sim.100fps"  
  echo $cmd
  eval $cmd

  ##this is the final processed file name
  trainingfile="${params[0]}-${params[1]}_trainingset.txt"
  echo $trainingfile

  #simstate_to_trainingset
  echo "Converting sim.100fps to trainingset"
  cmd="python ${scriptpath}simstate_to_trainingset.py sim.100fps ${trainingfile}"  
  echo $cmd
  eval $cmd

  #delete intermediate files
  echo "Removing intermediate data files"
  cmd="rm sim.100fps"
  echo $cmd
  eval $cmd
  cmd="rm sim.log"
  echo $cmd
  eval $cmd
  cmd="rm err.log"
  echo $cmd
  eval $cmd
  echo "Completed processing ${tarfile}" 
  echo ""
  #exit 1
done

