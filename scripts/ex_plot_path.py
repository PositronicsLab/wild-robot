"""
James Taylor

This example demonstrates plotting path data from processed training data.  The
script must be run in the directory where the data resides.

The script will also call a script that uses the imagemagick program to crop
the plot to consistent dimensions.  The crop script must be located in the same
directory as this python script.
"""
import numpy as np
import wbplot as wbplt
import os
import subprocess
import argparse

def get_timestep(data, column):
  return data[1,column] - data[0,column]

def calc_samples(maxtime, timestep):
  return int(float(maxtime) * 1/float(timestep))

def calc_arrowstep(arrowtime, timestep):
  return int(float(arrowtime) * 1/float(timestep))

def calc_decimation_factor(timestep, optimalstep):
  return int(optimalstep/timestep)

def decimate_data(data, step=10):
  idxs = np.arange(0,len(data),step)
  data = np.take(data, idxs, axis=0)
  return data

parser = argparse.ArgumentParser(description='Plot WB paths')
#parser.add_argument('timecolumn')
parser.add_argument('timestep')
parser.add_argument('--maxtime')
parser.add_argument('--arrowtime')
parser.add_argument('-c', '--color')
parser.add_argument('-x', '--pos', nargs=2)
parser.add_argument('-dx', '--vel', nargs=2)
parser.add_argument('--rot', choices=['0', '90', '180', '270'], default='0')
parser.add_argument('--flip', choices=['n','v','h'], default='n')
parser.add_argument('--imgfile')
#parser.add_argument('-s', '--save', choices=['n','y'], default='n')
parser.add_argument('infile')

args = parser.parse_args()
print args

save = True
if args.imgfile is None:
  save = False
else:
  imgfile = args.imgfile

#save = False
#if args.save =='y':
#  save = True
#print save

data = np.loadtxt(args.infile)

if args.pos is not None:
  poscols=(int(args.pos[0]),int(args.pos[1]))
else:
  poscols=(0,1)

if args.vel is not None:
  velcols=(int(args.vel[0]),int(args.vel[1]))
else:
  velcols=(0,1)

dims = (0,1)
sgns = (1,1)
#transforms are rotation first with rotations in the cw direction
if args.rot != '0':
  if args.rot == '90':
    dims = (1,0)
    sgns = (1,-1)
  elif args.rot == '180':
    dims = (0,1)
    sgns = (-1,-1)
  elif args.rot == '270':
    dims = (1,0)
    sgns = (-1,1)

if args.flip == 'h':
  sgns = (-sgns[0],sgns[1])
elif args.flip == 'v':
  sgns = (sgns[0],-sgns[1])

if args.color is not None:
  color = args.color
else:
  color = 'b'

#samples = calc_samples(args.maxtime, timestep)
#arrowstep = calc_arrowstep(args.arrowtime, timestep)

optstep = 0.01
#timestep = get_timestep(data, int(args.timecolumn))
timestep = float(args.timestep)
decfactor = calc_decimation_factor(timestep, optstep)
data = decimate_data(data, decfactor)
if args.maxtime is None:
  samples = data.shape[0]
else:
  samples = calc_samples(int(args.maxtime), optstep)
arrowstep = calc_arrowstep(args.arrowtime, optstep)

if samples > data.shape[0]:
  samples = data.shape[0]
x = data[:samples,poscols[0]:poscols[1]+1]
dx = data[:samples,velcols[0]:velcols[1]+1]

fig, ax = wbplt.pathplot_figure()
wbplt.pathplot_centerlines(fig, ax)
wbplt.pathplot_enclosure(fig, ax)
wbplt.pathplot_position(fig, ax, x, dims=dims, sgns=sgns, color=color)
wbplt.pathplot_arrows(fig, ax, x, dx, dims=dims, sgns=sgns, step=arrowstep, color=color)
wbplt.pathplot_endpoints(fig, ax, x, dims=dims, sgns=sgns, fill=color)
#wbplt.pathplot_subpaths(fig, ax, x, dx, dims=dims, sgns=sgns, step=100)

if not save:
  # aspect is likely skewed when rendered to window
  wbplt.show()
else:
  # aspect is preserved when rendered to file
  #sfx = sim + '_' + 'offset' + offset + '_' + str(int(t)) + 's'
  #imgfile = 'plot_path_' + sfx + '.png'

  scriptdir = os.path.dirname(os.path.abspath(__file__))
  ppscript = os.path.join(scriptdir,"crop_path_plot.sh")
  if os.path.isfile(ppscript):
    tmp = 'temp.png'
    wbplt.save(tmp)
    
    subprocess.call('bash ' + ppscript + ' %s %s' % (tmp,imgfile,), shell=True)
    subprocess.call('rm %s' % (tmp,), shell=True)
  else:
    print imgfile
    wbplt.save(imgfile)
