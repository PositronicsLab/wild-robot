import numpy as np
import re

#------------------------------------------------------------------------------
"""
Correct positions in the vicon data to center on the middle of the enclosure.
This is due to inexact positioning of the enclosure and/or the wand during
motion capture, so this is only necessary to perform on raw vicon data.
"""
def correct_position(x, dims=(1,2)):
  for dim in dims:
    mx = max(x[:,dim])
    mn = min(x[:,dim])
    x[:,dim] -= mx-(mx-mn)/2.0
  return x

"""
Parse position information from the data format used in the raw wb vicon files
"""
def parse_raw_vicon_position(line):
  a = re.split( ':', line )
  a = re.split( ',', a[1] )
  a = np.array(map(float, a))
  #print( a )
  return a

"""
Parse rotation information from the data format used in the raw wb vicon files
"""
def parse_raw_vicon_rotation(line):
  a = re.split( ':', line )
  a = re.split( ',', a[1] )
  a = np.array(map(float, a))
  #print( a )
  return a

"""
Read a raw wb vicon file and extract all state data as a numpy array
"""
def read(path, center=True):
  try:
    f = open(path)
  except Exception:
    return[]

  content = [x.strip('\r\n') for x in f.readlines() ]
  f.close()

  state = np.array([0,0,0,0,0,0,0,0])
  t = 0
  dt = 0.01

  i = 0
  for line in content:
    if i == 0:
      i = i + 1
    elif i == 1:
      i = i + 1
      pos = parse_raw_vicon_position(line)
    elif i == 2:
      i = i + 1
      rot = parse_raw_vicon_rotation(line)
      x = np.array([t, pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]])
      state = np.vstack([state, x])
    else:
      i = 0
      t = t + dt

  state = np.delete(state, (0), axis=0)
  if center:
    state = correct_position(state)
  return state

