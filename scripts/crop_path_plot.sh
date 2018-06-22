#!/bin/bash

### Crops whitespace from a path plot

infile=$1
outfile=$2

cmd="convert -crop 2400x2400+440+416 $infile $outfile"
echo $cmd
eval $cmd
