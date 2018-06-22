"""
James Taylor

This example demonstrates plotting path data from processed training data.  The
script must be run in the directory where the data resides.  The script parses
the name of the input file for particular tokens which are used to name the
output file, so this script must be run on files that follow the expected
naming convention
argument 1 is the path to the training data that is to be plotted
argument 2 is the length in seconds to be plotted

The script will also call a script that uses the imagemagick program to crop
the plot to consistent dimensions.  The crop script must be located in the same
directory as this python script.

Usage:
python ex_plot_training_path.py <path-to-training-data-file> <time>

Example (plot 120s of data from an ode training set):
python ../scripts/ex_plot_training_path.py ode-offset-0_trainingset.txt 120
"""
import numpy as np
import wb_raw_vicon_reader as raw_vicon_reader
import wbplot as wbplt
import sys
import os
import subprocess

save = True
path = sys.argv[1]
tokens = path.split('-')
sim = tokens[0]
tokens = path.split('_')
tokens2 = tokens[0][len(sim):].split('t')
offset = tokens2[1]

t = float(sys.argv[2])   # time in seconds
samples = int(t * 100)

#print path
#print sim
#print offset
#quit()

data = np.loadtxt(path)
if samples > data.shape[0]:
  samples = data.shape[0]
#print samples
#quit()
x = data[:samples,0:2]
dx = data[:samples,7:9]

dims=(0,1)
#dims=(1,2)
sgns=(1,1)
fig, ax = wbplt.pathplot_figure()
wbplt.pathplot_centerlines(fig, ax)
wbplt.pathplot_enclosure(fig, ax)
wbplt.pathplot_position(fig, ax, x, dims=dims, sgns=sgns, color='b')
wbplt.pathplot_arrows(fig, ax, x, dx, dims=dims, sgns=sgns, color='b')
wbplt.pathplot_endpoints(fig, ax, x, dims=dims, sgns=sgns, fill='b')
#wbplt.pathplot_subpaths(fig, ax, x, dx, dims=dims, sgns=sgns, step=100)

if not save:
  # aspect is likely skewed when rendered to window
  wbplt.show()
else:
  # aspect is preserved when rendered to file
  sfx = sim + '_' + 'offset' + offset + '_' + str(int(t)) + 's'
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
