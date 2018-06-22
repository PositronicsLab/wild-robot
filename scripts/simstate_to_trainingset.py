import numpy as np
#import re
#import math
import sys

infile=sys.argv[1]  #input file to process
outfile=sys.argv[2]  #output file

A=np.loadtxt(infile)
#print A
#print("shape:"+str(A.shape))

B = np.empty((0))
for i in range(len(A)-1):
  x0 = A[i]
  x1 = A[i+1]
  x = [x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x0[8], x0[9], x0[10], x0[11], x0[12], x0[13], x0[14], x0[15], x1[1], x1[2], x1[3], x1[4], x1[5], x1[6], x1[7], x1[8], x1[9], x1[10], x1[11], x1[12], x1[13], x1[14], x1[15]]
  B = np.append(B, x)
B=np.reshape(B, (len(B)/30,30))
#print B

np.savetxt(outfile, B, delimiter=" ", fmt="%1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e")

