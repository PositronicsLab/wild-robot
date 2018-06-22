"""
For sims that step at 10us and to reduce to 10ms samples an example command is:
python ../../../../scripts/decimate_state.py 999 1000 sim.log sim.100fps
"""

import numpy as np
import sys
#import time

def decimate_data(x, offset=0, step=10):
  idxs = np.arange(offset, len(x), step)
  x = np.take(x, idxs, axis=0)
  return x

offset=int(sys.argv[1])
step=int(sys.argv[2])
infile=sys.argv[3]  #input file to process
outfile=sys.argv[4]  #output file

A=np.loadtxt(infile)
#print A
#print "shape: "+str(A.shape)
A=decimate_data(A, offset, step)
#print "shape: "+str(A.shape)
#print A

np.savetxt(outfile, A, delimiter=" ", fmt="%1.8e")
