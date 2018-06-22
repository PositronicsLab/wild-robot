import h5py
import numpy as np
import sys
import os

def join(path, key):
  if path[-1] != '/':
    path += '/'
  return path + key

#def build_mocap_models(rootdir, h5file):
#  sgrp_root = 'mocap/models'
#  sds_wb = join(sgrp_root, 'wb.vsk')
#
#  print 'creating group: ' + sgrp_root
#  #h5file.create_group(sgrp_root)
#  print 'creating dataset: ' + sds_wb
#  #h5file.create_dataset(sds_wb,,dtype='f')

def build_mocap_sessions(rootdir, h5file, session):
  sgrp_parent = 'mocap/sessions'
  path_parent = os.path.join(os.path.join(rootdir, 'mocap'), 'sessions')
  sgrp_root = join(sgrp_parent, session)
  path_root = os.path.join(path_parent, session)
  sgrp_raw = join(sgrp_root, 'raw')
  path_raw = os.path.join(path_root, 'raw')
  sds_raw_state = join(sgrp_raw, 'state')
  path_raw_state = os.path.join(path_raw, 'state.txt')
  # Note: limits on git suggest video should not be integrated directly 
  #       into the dataset.  If the data is hosted through a different means
  #       video can be embedded, but for now, video is excluded and maintained
  #       as individual files
  #sds_raw_video = join(sgrp_raw, 'video')  
  sds_raw_signals = join(sgrp_raw, 'signals')
  sgrp_interp = join(sgrp_root, 'interpolated')
  sds_interp_state = join(sgrp_interp, 'state')

  print 'creating group: ' + sgrp_raw
  h5file.create_group(sgrp_raw)
  print 'creating dataset: ' + sds_raw_state
  arr = np.loadtxt(path_raw_state)
  #print arr
  ds = h5file.create_dataset(sds_raw_state, data=arr, compression='gzip')
  ds.attrs['fields'] = 't, shell(7){pos(x,y,z),rot(qx,qy,qz,qw)}'
  #ds.attrs['sample rate'] = 1e-2
  #print ds.attrs['fields']
  print 'creating dataset: ' + sds_raw_signals
  #h5file.create_dataset(sds_raw_signals,,dtype='f')
  print 'creating group: ' + sgrp_interp
  #h5file.create_group(sgrp_interp)
  print 'creating dataset: ' + sds_interp_state
  #h5file.create_dataset(sds_interp_state,,dtype='f')

def build_mocap_branch(rootdir, h5file):
  #build_mocap_models(rootdir, h5file)

  a = np.arange(10) + 1
  for i in a:
    session = str(i).zfill(2) 
    build_mocap_sessions(rootdir, h5file, session)

def build_simulation_gazebo_branch(rootdir, h5file):
  sims = ['ode','dart']
  sgrp_root = 'simulation/gazebo'
  for sim in sims:
    sgrp_sim = join(sgrp_root, sim)
    #sds_10us = join(sgrp_sim, 'step=10us')
    sds_state = join(sgrp_sim, 'state')

    print 'creating group: ' + sgrp_sim
    #h5file.create_group(sgrp_sim)
    #print 'creating dataset: ' + sds_10us
    #h5file.create_dataset(sds_10us,,dtype='f')
    print 'creating dataset: ' + sds_state
    arr = np.loadtxt(path_state)
    #print arr
    #ds = h5file.create_dataset(sds_state, data=arr, compression='gzip')
    #ds.attrs['step'] = 1e-5
    #ds.attrs['fields'] = 't,shell(13){pos(x,y,z),rot(qx,qy,qz,qw),lvel(dx,dy,dz),avel(omegax,omegay,omegaz)},joint(2){angle,vel}'
    #ds.attrs['sample rate'] = 1e-2

def build_simulation_simwise_branch(rootdir, h5file):
  sgrp_root = 'simulation/simwise4d'
  sds_1ms = join(sgrp_root, 'step=1ms')

  print 'creating group: ' + sgrp_root
  #h5file.create_group(sgrp_root)
  print 'creating dataset: ' + sds_1ms
  #h5file.create_dataset(sds_1ms,,dtype='f')
  #Note: should import the simulation file here as well

def build_simulation_branch(rootdir, h5file):
  build_simulation_gazebo_branch(rootdir, h5file)
  build_simulation_simwise_branch(rootdir, h5file)

#def build_models_branch(f):
#  sgrp_root = 'models'
#  #sds_shell = join(sgrp_root, '')
#
#  print 'creating group: ' + sgrp_root
#  #h5file.create_group(sgrp_root)
#  #print 'creating dataset: ' + sds_shell
#  #h5file.create_dataset(sds_shell,,dtype='f')

# root dir containing the filesystem hierarchy that maps the hdf5 structure
rootdir = sys.argv[1]      
h5fpath = sys.argv[2]
h5file = h5py.File(h5fpath, 'w')
#h5file = []

build_mocap_branch(rootdir, h5file)
build_simulation_branch(rootdir, h5file)
#build_models_branch(rootdir, h5file)

#print h5file.keys()
# print out all sessions
#for session in f[gsessions]:
#  print session

h5file.close()
