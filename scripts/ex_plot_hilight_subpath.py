"""
James Taylor

This example demonstrates plotting path data from raw vicon data with additional
highlighting of a particular subpath
argument 1 is the path to the vicon data that is to be plotted

Usage:
python ex_plot_hilight_subpath.py <path-to-file-containing-vicon-session-data>

Example:
python ex_plot_hilight_subpath.py ../../wild-robot/data/raw_experiment/vicon/trial_01.txt
"""
import numpy as np
import wb_raw_vicon_reader as raw_vicon_reader
import wbplot as wbplt
import sys

path = sys.argv[1]
i = 5

x = raw_vicon_reader.read(path)
dx = wbplt.sample_deltas(x)

dims=(1,2)
sgns=(1,1)

fig, ax = wbplt.pathplot_figure()
wbplt.pathplot_centerlines(fig, ax)
wbplt.pathplot_enclosure(fig, ax)
wbplt.pathplot_position(fig, ax, x, dims=dims, sgns=sgns)
wbplt.pathplot_arrows(fig, ax, x,dx, dims=dims, sgns=sgns)
wbplt.pathplot_endpoints(fig, ax, x, dims=dims, sgns=sgns)

idxs = wbplt.find_subpaths(x, dx, dims)
idx = idxs[i] 
wbplt.pathplot_position(fig, ax, x[idx[0]:idx[1],:], dims=dims, sgns=sgns, color='b')
wbplt.pathplot_arrows(fig, ax, x[idx[0]:idx[1],:], dx[idx[0]:idx[1],:], dims=dims, sgns=sgns, color='b')

wbplt.show()              # aspect is likely skewed when rendered to window
#wbplt.save('test.png')   # aspect is preserved when rendered to file
