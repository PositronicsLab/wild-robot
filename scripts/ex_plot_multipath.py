"""
James Taylor

This example demonstrates plotting path data from processed training data.  The
script must be run in the directory where the data resides.

The script will also call a script that uses the imagemagick program to crop
the plot to consistent dimensions.  The crop script must be located in the same
directory as this python script.

Note: this only works with tar.gz files that contain data named sim.log
"""
import numpy as np
import wbplot as wbplt
import os
import subprocess
import argparse

def get_timestep(data, column):
  return data[1,column] - data[0,column]

def calc_samples(maxtime, timestep):
  return int(float(maxtime) * 1.0/timestep)

def calc_arrowstep(arrowtime, timestep):
  return int(float(arrowtime) * 1/float(timestep))

def plot_path(fig, ax, x, dx, dims, sgns, color, arrowstep, label=None):
  wbplt.pathplot_position(fig, ax, x, dims=dims, sgns=sgns, color=color, label=label)
  wbplt.pathplot_arrows(fig, ax, x, dx, dims=dims, sgns=sgns, step=arrowstep, color=color)
  wbplt.pathplot_endpoints(fig, ax, x, dims=dims, sgns=sgns, fill=color)

def calc_decimation_factor(timestep, optimalstep):
  return int(optimalstep/timestep)

def decimate_data(data, step=10):
  idxs = np.arange(0,len(data),step)
  data = np.take(data, idxs, axis=0)
  return data

parser = argparse.ArgumentParser(description='Plot WB paths')
parser.add_argument('timecolumn')
parser.add_argument('--maxtime')
parser.add_argument('--arrowtime')
parser.add_argument('-x', '--pos', nargs=2)
parser.add_argument('-dx', '--vel', nargs=2)
parser.add_argument('--rot', choices=['0', '90', '180', '270'], default='0')
parser.add_argument('--flip', choices=['n','v','h'], default='n')
parser.add_argument('--imgfile')
parser.add_argument('simulator')

args = parser.parse_args()
sims = ['ode','dart','bullet']
if args.simulator not in sims:
  print 'simulator argument is not valid.  simulator must be one of ' + str(sims)
  exit(1)
print args

save = True
if args.imgfile is None:
  save = False
else:
  imgfile = args.imgfile

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

sim = args.simulator
optstep = 0.01
maxtime = int(args.maxtime)

fig, ax = wbplt.pathplot_figure()

wbplt.pathplot_centerlines(fig, ax)
wbplt.pathplot_enclosure(fig, ax)

arcname = sim + '_1ms.tar.gz'
subprocess.call('/bin/tar -xvzf %s' % (arcname,), shell=True)
fname = 'sim.log'
data = np.loadtxt(fname)
color = 'r'
timestep = get_timestep(data, int(args.timecolumn))
decfactor = calc_decimation_factor(timestep, optstep)
data = decimate_data(data, decfactor)
samples = calc_samples(maxtime, optstep)
arrowstep = calc_arrowstep(args.arrowtime, optstep)
if samples > data.shape[0]:
  samples = data.shape[0]
x = data[:samples,poscols[0]:poscols[1]+1]
dx = data[:samples,velcols[0]:velcols[1]+1]
plot_path(fig, ax, x, dx, dims, sgns, color, arrowstep, label='1ms')

arcname = sim + '_100us.tar.gz'
subprocess.call('/bin/tar -xvzf %s' % (arcname,), shell=True)
fname = 'sim.log'
data = np.loadtxt(fname)
color = 'g'
timestep = get_timestep(data, int(args.timecolumn))
decfactor = calc_decimation_factor(timestep, optstep)
data = decimate_data(data, decfactor)
samples = calc_samples(maxtime, optstep)
arrowstep = calc_arrowstep(args.arrowtime, optstep)
print arrowstep
if samples > data.shape[0]:
  samples = data.shape[0]
x = data[:samples,poscols[0]:poscols[1]+1]
dx = data[:samples,velcols[0]:velcols[1]+1]
plot_path(fig, ax, x, dx, dims, sgns, color, arrowstep, label='100$\mu$s')

arcname = sim + '_10us.tar.gz'
subprocess.call('/bin/tar -xvzf %s' % (arcname,), shell=True)
fname = 'sim.log'
data = np.loadtxt(fname)
color = 'b'
timestep = get_timestep(data, int(args.timecolumn))
decfactor = calc_decimation_factor(timestep, optstep)
data = decimate_data(data, decfactor)
samples = calc_samples(maxtime, optstep)
arrowstep = calc_arrowstep(args.arrowtime, optstep)
if samples > data.shape[0]:
  samples = data.shape[0]
x = data[:samples,poscols[0]:poscols[1]+1]
dx = data[:samples,velcols[0]:velcols[1]+1]
plot_path(fig, ax, x, dx, dims, sgns, color, arrowstep, label='10$\mu$s')

ax.legend(loc=1,bbox_to_anchor=(0.9, 0.9), fontsize='xx-large')

if not save:
  # aspect is likely skewed when rendered to window
  wbplt.show()
else:
  # aspect is preserved when rendered to file

  scriptdir = os.path.dirname(os.path.abspath(__file__))
  ppscript = os.path.join(scriptdir,"crop_path_plot.sh")
  if os.path.isfile(ppscript):
    tmp = 'temp.png'
    wbplt.save(tmp)
    
    subprocess.call('bash ' + ppscript + ' %s %s' % (tmp,imgfile,), shell=True)
    subprocess.call('rm %s' % (tmp,), shell=True)
  else:
    wbplt.save(imgfile)
