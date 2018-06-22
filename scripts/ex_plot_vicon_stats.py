"""
This example demonstrates plotting frequency data from processed vicon data
which must contain interpolated motor frequency 
argument 1 is the path to the vicon data that is to be plotted
argument 2 is a numeric session identifier which is used as a suffix in the plot
           output file.

Usage:
python ex_plot_vicon_stats.py <path-to-file-containing-vicon-session-data> <session-id>

Example:
python ex_plot_vicon_stats.py ../../wild-robot/data/raw_experiment/vicon/trial_01.txt 1
"""
import numpy as np
import wb_raw_vicon_reader as raw_vicon_reader
import wbplot as wbplt
import math
import sys

#------------------------------------------------------------------------------
"""
python optimal and numpy compatible convenience method to compute the angle 
between two quaterions
"""
def quat_angle( q1, q2 ):
  dot = q1[:,0]*q2[:,0] + q1[:,1]*q2[:,1] + q1[:,2]*q2[:,2] + q1[:,3]*q2[:,3];
  dot[dot<-1] = -1
  dot[dot>1] = 1
  return np.arccos(abs(dot));

#------------------------------------------------------------------------------
path = sys.argv[1]
session_id = int(sys.argv[2])
session_sfx = str(session_id).zfill(2)

dat = raw_vicon_reader.read(path)
t = dat[:,0]
x = dat[:,1:4]
q = dat[:,4:8]
dx = wbplt.sample_deltas(dat)
dq = quat_angle(q[:-1,:],q[1:,:])

color_actual_plot = '#2080b0'
color_actual_mu = '#0000ff'
color_desired_mu = '#ff0000'
color_desired_plot = '#ff0000'
hilit_desired_mu = '#ffb0b0'

wb_r = 0.041

mu = x[:,2].mean()
fig, ax = wbplt.mocapplot_figure()
wbplt.mocapplot_z(fig, ax, t, x[:,2], mu, wb_r)
imgname = 'plot_z_' + session_sfx + '.png'
#wbplt.save(imgname)
wbplt.show()

mu = x[:,2].mean()
fig, ax = wbplt.mocapplot_figure()
wbplt.mocapplot_z_histogram(fig, ax, x[:,2], mu, wb_r)
imgname = 'plot_z_hist_' + session_sfx + '.png'
#wbplt.save(imgname)
wbplt.show()

figcache, axcache = wbplt.mocapplot_figure()
n, bins, patches = wbplt.mocapplot_z_distribution_cache(figcache, axcache, x[:,2])
mu = x[:,2].mean()
fig, ax = wbplt.mocapplot_figure()
wbplt.mocapplot_z_distribution(fig, ax, x[:,2], mu, wb_r, n, bins, patches)
imgname = 'plot_z_gaus_' + session_sfx + '.png'
#wbplt.save(imgname)
wbplt.show()

mu = dq.mean()
fig, ax = wbplt.mocapplot_figure()
wbplt.mocapplot_dtheta(fig, ax, t[:-1], dq, mu)
imgname = 'plot_dtheta_' + session_sfx + '.png'
#wbplt.save(imgname)
wbplt.show()

