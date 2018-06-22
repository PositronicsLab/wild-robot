"""
James Taylor

This example demonstrates plotting path data from raw vicon data.
argument 1 is the path to the vicon data that is to be plotted
argument 2 is a numeric session identifier which is used as a suffix in the plot
           output file.

Usage:
python ex_plot_vicon_path.py <path-to-file-containing-vicon-session-data> <session-id>

Example:
python ex_plot_vicon_path.py ../../wild-robot/data/raw_experiment/vicon/trial_01.txt 1
"""
import numpy as np
import wb_raw_vicon_reader as raw_vicon_reader
import wbplot as wbplt
import sys
import os
import subprocess

color = '#00a0ff'
save = True

path = sys.argv[1]
session = int(sys.argv[2])

x = raw_vicon_reader.read(path)
dx = wbplt.sample_deltas(x)

dims=(1,2)
sgns=(1,1)
fig, ax = wbplt.pathplot_figure()
wbplt.pathplot_centerlines(fig, ax)
wbplt.pathplot_enclosure(fig, ax)
wbplt.pathplot_position(fig, ax, x, dims=dims, sgns=sgns, color=color)
wbplt.pathplot_arrows(fig, ax, x, dx, dims=dims, sgns=sgns, color=color)
wbplt.pathplot_endpoints(fig, ax, x, dims=dims, sgns=sgns, fill=color)
#wbplt.pathplot_subpaths(fig, ax, x, dx, dims=dims, sgns=sgns, step=100)

if not save:
  # aspect is likely skewed when rendered to window
  wbplt.show()
else:
  # aspect is preserved when rendered to file
  sfx = str(session).zfill(2)
  img = 'plot_path_' + sfx + '.png'

  scriptdir = os.path.dirname(os.path.abspath(__file__))
  ppscript = os.path.join(scriptdir,"crop_path_plot.sh")
  if os.path.isfile(ppscript):
    tmp = 'temp.png'
    wbplt.save(tmp)
    
    subprocess.call('bash ' + ppscript + ' %s %s' % (tmp,img,), shell=True)
    subprocess.call('rm %s' % (tmp,), shell=True)
  else:
    print img
    wbplt.save(img)
