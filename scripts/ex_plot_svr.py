"""
James Taylor

This example demonstrates plotting svr analysis.
"""
import numpy as np
import wbplot as wbplt
import sys

#def plot_fit_and_error(fig, ax, alg, offsets, sim, color, color2, markersz=4):
  

#------------------------------------------------------------------------------
"""
"""
def plot_training_error(fig, ax, alg, offsets, sim, color, color2, markersz=4):
  y_pred = np.loadtxt( 'svr_' + sim + '_' + alg + '_y-fit_training.txt')
  y_test = np.loadtxt( 'svr_' + sim + '_y-train.txt')

  # TODO: the limit on training error needs to be closer to the offset so that the error line is more visible
  lims=([0,len(y_pred)],[0,0.0175])
  #lims=([0,len(y_pred)],[0,0.01])

  data = [{'y-test': y_test, 'y-pred': y_pred, 'label':sim.upper(), 'color':color, 'color2':color2}]
  
  wbplt.svrplot_training_error(fig, ax, alg, data, offsets, markersz, lims=lims)

#------------------------------------------------------------------------------
"""
"""
def plot_testing_error(fig, ax, alg, offsets, sim, color, color2, markersz=4):
  y_pred = np.loadtxt( 'svr_' + sim + '_' + alg + '_y-fit_testing.txt')
  y_test = np.loadtxt( 'svr_' + sim + '_y-test.txt')

  lims=([0,len(y_pred)],[0,0.0175])
  #lims=([0,len(y_pred)],[0,0.01])

  data = [{'y-test': y_test, 'y-pred': y_pred, 'label':sim.upper(), 'color':color, 'color2':color2}]
  
  wbplt.svrplot_testing_error(fig, ax, alg, data, offsets, markersz, lims=lims)
  
#------------------------------------------------------------------------------
"""
"""
def plot_fit(fig, ax, alg, offsets, sim, trial, color, color2, markersz=4):
#def plot_fit(alg, offsets, trial, sim):
  pfx = alg + '_' + sim + '_trial-' + str(trial).zfill(2) + '_' 
  y_fit = np.loadtxt( pfx + 'y-fit.txt')
  
  data = [{'y-fit': y_fit, 'label':sim.upper(), 'color':color, 'color2':color2}]
    
  wbplt.svrplot_fit(fig, ax, alg, data, offsets)

#------------------------------------------------------------------------------
"""
"""
def plot_session(alg, sims, offsets, trial):
  data = {}
  for sim in sims:
    infile = 'svr_' + sim + '_' + alg + '_y-fit_session-' + str(trial).zfill(2) + '.txt' 
    y_fit = np.loadtxt(infile);

    if sim == 'ode':
      data['ode'] = [{'y-fit': y_fit, 'label':'ODE', 'color':'b', 'color2':'c'}]
    elif sim == 'dart':
      data['dart'] = [{'y-fit': y_fit, 'label':'DART', 'color':'r', 'color2':'m'}]
  
  fig, ax = wbplt.svrplot_figure()
  for sim in sims:
    wbplt.svrplot_fit_line(fig, ax, alg, trial, data.get(sim), offsets)
  outfile = 'svr_' + alg + '_session-' + str(trial).zfill(2) + '_samples.png'
  wbplt.save(outfile)
  #wbplt.show()
  wbplt.close(fig)

  fig, ax = wbplt.svrplot_figure()
  for sim in sims:
    wbplt.svrplot_fit_histogram(fig, ax, alg, trial, data.get(sim), offsets)
  outfile = 'svr_' + alg + '_session-' + str(trial).zfill(2) + '_histogram.png'
  wbplt.save(outfile)
  #wbplt.show()
  wbplt.close(fig)

#------------------------------------------------------------------------------
"""
"""
def plot_group(alg, sims, offsets, trials):
  data = {}
  for sim in sims:
    i = 0
    for trial in trials:
      infile = 'svr_' + sim + '_' + alg + '_y-fit_session-' + str(trial).zfill(2) + '.txt' 
      if i == 0:
        y_fit = np.loadtxt(infile);
      else:
        y_fit = np.append(y_fit,np.loadtxt(infile),axis=0)
      i += 1
  
    if sim == 'ode':
      data['ode'] = [{'y-fit': y_fit, 'label':'ODE', 'color':'b', 'color2':'c'}]
    elif sim == 'dart':
      data['dart'] = [{'y-fit': y_fit, 'label':'DART', 'color':'r', 'color2':'m'}]
  
  #data = [ode, dart]
  
  label = str(trials[0]) + '-' + str(trials[-1])
  s = str(trials[0]).zfill(2) + '-' + str(trials[-1]).zfill(2)
  
  fig, ax = wbplt.svrplot_figure()
  for sim in sims:
    wbplt.svrplot_fit_line(fig, ax, alg, label, data.get(sim), offsets)
  outfile = 'svr_' + alg + '_sessions_' + s + '_samples.png'
  wbplt.save(outfile)
  #wbplt.show()
  wbplt.close(fig)
  
  fig, ax = wbplt.svrplot_figure()
  for sim in sims:
    wbplt.svrplot_fit_histogram(fig, ax, alg, label, data.get(sim), offsets)
  outfile = 'svr_' + alg + '_sessions_' + s + '_histogram.png'
  wbplt.save(outfile)
  #wbplt.show()
  wbplt.close(fig)

#------------------------------------------------------------------------------
algs=['rbf','linearsvr']
sims = ['dart','ode']
offsets = [-0.013248,-0.009936,-0.006624,-0.003312, 0.0, 0.003312, 0.006624, 0.009936, 0.013248]
simdata = {'ode':{'color':'b','color2':'c','label':'ODE'},'dart':{'color':'r','color2':'m','label':'DART'}}

alg = sys.argv[1]
sim = sys.argv[2]

if alg not in algs:
  print 'arg[1] algorithm is not valid.  algorithm must be one of ' + str(algs)
  exit(1)

if sim not in sims:
  print 'arg[2] simulator is not valid.  simulator must be one of ' + str(sims)
  exit(1)

#sims = ['dart']
#sims = ['ode']
#sim = sims[0]

sims = [sim]

d = simdata.get(sim)
fig, ax = wbplt.svrplot_figure()
plot_testing_error(fig, ax, alg, offsets, sim, d.get('color'), d.get('color2'), markersz=4)
imgfile = 'svr_' + sim + '_' + alg + '_testing-error.png'
wbplt.save(imgfile)
#wbplt.show()
wbplt.close(fig)

fig, ax = wbplt.svrplot_figure()
plot_training_error(fig, ax, alg, offsets, sim, d.get('color'), d.get('color2'), markersz=4)
imgfile = 'svr_' + sim + '_' + alg + '_training-error.png'
wbplt.save(imgfile)
#wbplt.show()
wbplt.close(fig)

#for sim in sims:
#  d = simdata.get(sim)
#  fig, ax = wbplt.svrplot_figure()
##  plot_training_error(fig, ax, alg, offsets, sim, d.get('color'), d.get('color2'), markersz=4)
#  plot_testing_error(fig, ax, alg, offsets, sim, d.get('color'), d.get('color2'), markersz=4)
#  #wbplt.show()
#  outfile = 'svr-' + alg + '-' + sim + '-training_error.png'
#  wbplt.save(outfile)
#  wbplt.close(fig)
#
#fig, ax = wbplt.svrplot_figure()
#for sim in sims:
#  d = simdata.get(sim)
##  plot_training_error(fig, ax, alg, offsets, sim, d.get('color'), d.get('color2'), markersz=4)
#  plot_testing_error(fig, ax, alg, offsets, sim, d.get('color'), d.get('color2'), markersz=4)
##wbplt.show()
#outfile = 'svr-' + alg + '-training_error.png'
#wbplt.save(outfile)
#wbplt.close(fig)
#


#sessions = [1,2,3,4,5,6,7,8,9,10]
sessions = [1]
#
plot_group(alg, sims, offsets, sessions)
for session in sessions:
  plot_session(alg, sims, offsets, session)
# 
#sessions = [1,2,3,4,5,6]
#plot_group(alg, sims, offsets, sessions)
#sessions = [7,8,9,10]
#plot_group(alg, sims, offsets, sessions)



#fig, ax = wbplt.svrplot_figure()
#for sim in sims:
#  trial = 1
#  d = simdata.get(sim)
#  #plot_testing_error(fig, ax, alg, offsets, sim, d.get('color'), d.get('color2'), markersz=4)
#  #plot_training_error(fig, ax, alg, offsets, sim, d.get('color'), d.get('color2'), markersz=4)
#  plot_fit(fig, ax, alg, offsets, sim, trial, d.get('color'), d.get('color2'), markersz=4)
#
#wbplt.show()
##outfile = 'svr-' + alg + '-' + sim + '-fit_and_error.png'
##wbplt.save(outfile)
#wbplt.close(fig)


