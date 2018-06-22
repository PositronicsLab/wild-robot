import h5py
import numpy as np
import sys

gsessions = 'mocap/session'
ggazebo = 'simulation/gazebo'
gsimwise = 'simulation/simwise'

fpath = sys.argv[1]

f = h5py.File(fpath, 'r')

#print f.keys()
# print out all sessions
for session in f[gsessions]:
  print session

f.close()
