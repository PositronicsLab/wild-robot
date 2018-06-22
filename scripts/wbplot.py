"""
James Taylor

Module generates a number of plots for the weazelball study
See python scripts prefixed with 'ex_plot' for examples of plotting data
"""

import matplotlib
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
from matplotlib.patches import Circle
import numpy as np

#------------------------------------------------------------------------------
"""
Returns a subset of the data given a step size
"""
def subsample_data( a, step ):
  return a[0::step,:]

#------------------------------------------------------------------------------
"""
Returns the first and last elements in a data set
"""
def subsample_endpts( a ):
  return np.array([a[0,:],a[-1,:]])

#------------------------------------------------------------------------------
"""
Computes the difference between samples across a data set
"""
def sample_deltas( a ):
  return np.diff(a, n=1, axis=0)

#------------------------------------------------------------------------------
"""
Saves a figure
"""
def save(savefile, dpi=200, transparent=True):
    plt.savefig(savefile, dpi=dpi, transparent=transparent)

#------------------------------------------------------------------------------
"""
Renders a figure to the onscreen renderer and shows it
"""
def show():
    plt.show()

#------------------------------------------------------------------------------
"""
Close a figure.  May be necessary when rendering a large number of figures, but
for only a few figures, it is not needed
"""
def close(fig):
    plt.close(fig)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
"""
Creates a new enclosure state figure which generally draws position data 
relative to the enclosure.  Must be called by client script if rendering onto a
single figure; otherwise, each drawing function renders onto its own figure
"""
def pathplot_figure():
  fig, ax = plt.subplots(figsize=(16,16)) 
  plt.axis('off')
  ax.set_xlim([-0.6,0.6])
  ax.set_ylim([-0.6,0.6])
  return fig, ax 

#------------------------------------------------------------------------------
"""
Compute set of arrow positions and directions along a path
"""
def compute_arrows(q, dq, step=100, dims=(0,1)):
  x = subsample_data( q, step )
  d = subsample_data( dq, step )
  l2n = np.linalg.norm( d[:,[dims[0],dims[1]]], axis=1, ord=2 )
  d[:,dims[0]] /= l2n
  d[:,dims[1]] /= l2n
  return x, d

#------------------------------------------------------------------------------
"""
Locates all subpaths within a data set.
"""
def find_subpaths( x, dx, dims=(0,1) ):
  max0 = max(x[:,dims[0]])
  max1 = max(x[:,dims[1]])
  tracking = False
  start_idx = -1
  idxs = np.array([])
  first = True
  for i in range(len(x)-1):
    if not tracking:
      if abs(x[i,dims[0]]) < max0 - 0.001 and \
         abs(x[i,dims[1]]) < max1 - 0.001:
        if np.sign(x[i,dims[0]]) != np.sign(dx[i,dims[0]]) and \
           np.sign(x[i,dims[1]]) != np.sign(dx[i,dims[1]]):
          tracking = True
          start_idx = i
    else:
      if abs(dx[i,dims[0]]) < 0.001 and \
         abs(dx[i,dims[1]]) < 0.001:
        if i - start_idx > 50:
          if first:
            idxs = np.array([start_idx,i])
            first = False
          else:
            idxs = np.vstack([idxs,np.array([start_idx,i])])
        tracking = False
        start_idx = -1
  return idxs

#------------------------------------------------------------------------------
"""
Draws enclosure center line axes onto a figure
"""
def pathplot_centerlines(fig, ax, color='g'):
  hdl, = ax.plot( [0,0], [-0.6,0.6], color=color, linewidth=1 )
  hdl, = ax.plot( [-0.6,0.6], [0,0], color=color, linewidth=1 )

#------------------------------------------------------------------------------
"""
Draws enclosure extens onto a figure
"""
def pathplot_enclosure(fig, ax, color='g'):
  o = 0.557    # outer box radius
  i = 0.516    # inner box radius
  hdl, = ax.plot( [-o,-o], [-o,o], color=color, linewidth=1 )
  hdl, = ax.plot( [-o,o], [o,o], color=color, linewidth=1 )
  hdl, = ax.plot( [o,o], [o,-o], color=color, linewidth=1 )
  hdl, = ax.plot( [o,-o], [-o,-o], color=color, linewidth=1 )
  hdl, = ax.plot( [-i,-i], [-i,i], color=color, linewidth=1, ls='--' )
  hdl, = ax.plot( [-i,i], [i,i], color=color, linewidth=1, ls='--' )
  hdl, = ax.plot( [i,i], [i,-i], color=color, linewidth=1, ls='--' )
  hdl, = ax.plot( [i,-i], [-i,-i], color=color, linewidth=1, ls='--' )

#------------------------------------------------------------------------------
"""
Draws wb state position onto a figure
"""
def pathplot_position(fig, ax, x, dims=(0,1), sgns=(1,1), color='r', Linewidth=4, label=None):
  hdl, = ax.plot( sgns[0]*x[:,dims[0]], sgns[1]*x[:,dims[1]], color=color, linewidth=Linewidth, label=label )

#------------------------------------------------------------------------------
"""
Draws wb direction arrows onto a figure
"""
def pathplot_arrows(fig, ax, x, dx, dims=(0,1), sgns=(1,1), color='r', step=100, scale=100, headwidth=10):
  a, da = compute_arrows(x, dx, step=step, dims=dims)

  ax.quiver( sgns[0]*a[:,dims[0]], sgns[1]*a[:,dims[1]], sgns[0]*da[:,dims[0]], sgns[1]*da[:,dims[1]], color=color, scale=scale, headwidth=headwidth )
  ax.scatter( sgns[0]*a[:,dims[0]], sgns[1]*a[:,dims[1]], color=color )

#------------------------------------------------------------------------------
"""
Draws wb startpoint onto a figure with size that reflects the true dimensions of
the wb.  The start point has a solid edge.
"""
def pathplot_startpoint(fig, ax, x, dims=(0,1), sgns=(1,1), fill='r', edge='k'):
  x = subsample_endpts(x)
  d = 0.084       # wb diameter
  r = d/2.0       # wb radius
  alpha = 0.25
  lw = 4
  e = Ellipse((sgns[0]*x[0,dims[0]], sgns[1]*x[0,dims[1]]),d, d, color=fill, zorder=100)
  e.set_alpha(alpha)
  ax.add_artist(e)
  c = Circle((sgns[0]*x[0,dims[0]], sgns[1]*x[0,dims[1]]),r, ec=edge, fc='none', lw = lw, zorder=100)
  ax.add_artist(c)
  #e = Ellipse((sgns[0]*x[1,dims[0]], sgns[1]*x[1,dims[1]]),d, d, color=fill, zorder=100)
  #e.set_alpha(alpha)
  #ax.add_artist(e)
  #c = Circle((sgns[0]*x[1,dims[0]], sgns[1]*x[1,dims[1]]),r, ec=edge, fc='none', lw = lw, ls='--', zorder=100)
  #ax.add_artist(c)

#------------------------------------------------------------------------------
"""
Draws wb endpoint onto a figure with size that reflects the true dimensions of
the wb.  The finish point has a dashed edge
"""
def pathplot_endpoint(fig, ax, x, dims=(0,1), sgns=(1,1), fill='r', edge='k'):
  x = subsample_endpts(x)
  d = 0.084       # wb diameter
  r = d/2.0       # wb radius
  alpha = 0.25
  lw = 4
  #e = Ellipse((sgns[0]*x[0,dims[0]], sgns[1]*x[0,dims[1]]),d, d, color=fill, zorder=100)
  #e.set_alpha(alpha)
  #ax.add_artist(e)
  #c = Circle((sgns[0]*x[0,dims[0]], sgns[1]*x[0,dims[1]]),r, ec=edge, fc='none', lw = lw, zorder=100)
  #ax.add_artist(c)
  e = Ellipse((sgns[0]*x[1,dims[0]], sgns[1]*x[1,dims[1]]),d, d, color=fill, zorder=100)
  e.set_alpha(alpha)
  ax.add_artist(e)
  c = Circle((sgns[0]*x[1,dims[0]], sgns[1]*x[1,dims[1]]),r, ec=edge, fc='none', lw = lw, ls='--', zorder=100)
  ax.add_artist(c)

#------------------------------------------------------------------------------
"""
Draws wb endpoints onto a figure with size that reflects the true dimensions of
the wb.  The start point has a solid edge and the finish point has a dashed edge
"""
def pathplot_endpoints(fig, ax, x, dims=(0,1), sgns=(1,1), fill='r', edge='k'):
  pathplot_startpoint(fig, ax, x, dims, sgns, fill, edge)
  pathplot_endpoint(fig, ax, x, dims, sgns, fill, edge)
#  x = subsample_endpts(x)
#  d = 0.084       # wb diameter
#  r = d/2.0       # wb radius
#  alpha = 0.25
#  lw = 4
#  #ax.scatter( sgns[0]*x[:,dims[0]], sgns[1]*x[:,dims[1]], color=fill, s=1 )
#  e = Ellipse((sgns[0]*x[0,dims[0]], sgns[1]*x[0,dims[1]]),d, d, color=fill, zorder=100)
#  e.set_alpha(alpha)
#  ax.add_artist(e)
#  c = Circle((sgns[0]*x[0,dims[0]], sgns[1]*x[0,dims[1]]),r, ec=edge, fc='none', lw = lw, zorder=100)
#  ax.add_artist(c)
#  e = Ellipse((sgns[0]*x[1,dims[0]], sgns[1]*x[1,dims[1]]),d, d, color=fill, zorder=100)
#  e.set_alpha(alpha)
#  ax.add_artist(e)
#  c = Circle((sgns[0]*x[1,dims[0]], sgns[1]*x[1,dims[1]]),r, ec=edge, fc='none', lw = lw, ls='--', zorder=100)
#  ax.add_artist(c)

#------------------------------------------------------------------------------
"""
Draws all wb subpaths onto a figure.  If a particular subpath desired, then this
can be used as a model in the main program
"""
def pathplot_subpaths(fig, ax, x, dx, dims=(0,1), sgns=(1,1), color='b', step=100, scale=100, headwidth=10):
  idxs = find_subpaths(x,dx,dims)

  for idx in idxs:
    pathplot_position(fig, ax, x[idx[0]:idx[1],:], dims=dims, sgns=sgns, color=color)

    pathplot_arrows(fig, ax, x[idx[0]:idx[1],:], dx[idx[0]:idx[1],:], dims=dims, sgns=sgns, color=color, step=step, scale=scale, headwidth=headwidth)

    ## Note: this is retained because it should go into an example of saving
    ##       all subpaths
    #if savefilepfx != '':
    #  savefile = savefilepfx + '_' + str(idx[0]+1).zfill(5) + '_' + str(idx[1]+1).zfill(5) + '.png'
    #  save(savefile)

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
"""
Creates a new matplotlib figure for svr analyses.  When renderering svr plots,
this function should be used to initialize the figure
"""
def svrplot_figure():
  fig, ax = plt.subplots() 
  return fig, ax 

#------------------------------------------------------------------------------
"""
Plots the values at each sample returned by svr as a result of fitting
"""
def svrplot_fit_line(fig, ax, alg, session, data, offsets):
  lw = 1

  for d in data:
    y = d.get('y-fit')
    x = np.arange(len(y))

    ax.plot(x, y, color=d.get('color'), lw=lw, label=d.get('label'))
    #ax.plot(x, y, color=d.get('color'), lw=lw, label=d.get('label'), marker='o', ls='')

  x = ax.get_xlim()

  for y in offsets:
    if y == 0:
      col = 'green'
    else:
      col = 'black'
    ax.plot( x, (y,y), color=col, lw=lw )

  ax.set_xlabel('Sample')
  ax.set_ylabel('Offset')
  title = 'Predicted WB CoM Offset - ' + alg.upper() + ' - Session ' + str(session)
  ax.set_title(title)
  ax.legend(loc=1)

#------------------------------------------------------------------------------
"""
Plots the aggregation of samples returned by svr as a result of fitting
"""
def svrplot_fit_histogram(fig, ax, alg, session, data, offsets, bins=50):
  x = []
  colors = []
  labels = []

  for d in data:
    x.append(d.get('y-fit'))
    colors.append(d.get('color'))
    labels.append(d.get('label'))

  ax.hist(x, bins, color=colors, label=labels)

  y = ax.get_ylim()
  for x in offsets:
    if x == 0:
      col = 'green'
    else:
      col = 'black'
    ax.plot( (x,x), y, color=col, linewidth=1 )
  ax.set_xlabel('Offset')
  ax.set_ylabel('Samples')
  title = 'Predicted WB CoM Offset - ' + alg.upper() + ' - Session ' + str(session)
  ax.set_title(title)
  ax.legend(loc=1)

  ax.title.set_fontsize(16)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)
  #ax.yaxis.get_major_formatter().set_powerlimits((0, 1))

  left = offsets[0] * (1 + 0.25)
  right = offsets[len(offsets)-1] * (1 + 0.25)
  ax.set_xlim([left,right])

#------------------------------------------------------------------------------
"""
Plot the training error for the cross validation test set on a given simulator
"""
def svrplot_training_error(fig, ax, alg, data, offsets, markersz=2, lims=(),legend=True):
  lw = 1

  for d in data:
    err = d.get('y-pred') - d.get('y-test')
    abserr = abs(err)
    d['mu'] = np.mean(abserr)
    d['std'] = np.std(abserr)

    x =np.arange(len(abserr))
    y = abserr 
    ax.scatter(x, y, color=d.get('color'), marker='o', s=8, label=d.get('label'))
    #ax.plot(x, y, color=d.get('color'), marker='o', ls='', label=d.get('label'), markersize=markersz)
    
  x1 = ax.get_xlim()

  for d in data:
    mu = d['mu']
    ax.plot( x1, (mu,mu), color=d.get('color2'), lw=lw )
    print 'Training Error(' + d.get('label') + '):' + str(mu) + ', ' + str(mu/offsets[5]*100) + '%'

  i = len(offsets)/2 + 1
  y = offsets[i]
  ax.plot( x1, (y,y), color='black', lw=lw )

  ax.set_xlabel('Sample')
  ax.set_ylabel('Absolute Error')
  title = 'Training Error - ' + alg.upper()
  ax.set_title(title)
  #ax.autoscale(True, axis='y', tight=False)
  if legend:
    ax.legend(loc=1)

  ax.title.set_fontsize(20)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)
  ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
  if lims:
    ax.set_ylim(lims[1])
    ax.set_xlim(lims[0])

#------------------------------------------------------------------------------
"""
Plot the testing error for the cross validation test set on a given simulator
"""
def svrplot_testing_error(fig, ax, alg, data, offsets, markersz=2, lims=(),legend=True):
  lw = 1

  for d in data:
    err = d.get('y-pred') - d.get('y-test')
    abserr = abs(err)
    d['mu'] = np.mean(abserr)
    d['std'] = np.std(abserr)

    x =np.arange(len(abserr))
    y = abserr 
    ax.scatter(x, y, color=d.get('color'), marker='o', s=8, label=d.get('label'))
    #ax.plot(x, y, color=d.get('color'), marker='o', ls='', label=d.get('label'), markersize=markersz)
    
  x1 = ax.get_xlim()

  for d in data:
    mu = d['mu']
    ax.plot( x1, (mu,mu), color=d.get('color2'), lw=lw )
    print 'Testing Error(' + d.get('label') + '):' + str(mu) + ', ' + str(mu/offsets[5]*100) + '%'

  i = len(offsets)/2 + 1
  y = offsets[i]
  ax.plot( x1, (y,y), color='black', lw=lw )

  ax.set_xlabel('Sample')
  ax.set_ylabel('Absolute Error')
  title = 'Testing Error - ' + alg.upper()
  ax.set_title(title)
  #ax.autoscale(True, axis='y', tight=False)
  if legend:
    ax.legend(loc=1)

  ax.title.set_fontsize(20)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)
  ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
  if lims:
    ax.set_ylim(lims[1])
    ax.set_xlim(lims[0])

#------------------------------------------------------------------------------
"""
"""
def svrplot_fit(fig, ax, alg, data, offsets, markersz=2, lims=(),legend=True):
  lw = 1

  for d in data:
    err = d.get('y-fit')
    abserr = abs(err)
    d['mu'] = np.mean(abserr)
    d['std'] = np.std(abserr)

    x =np.arange(len(abserr))
    y = abserr 
    ax.scatter(x, y, color=d.get('color'), marker='o', s=8, label=d.get('label'))
    #ax.plot(x, y, color=d.get('color'), marker='o', ls='', label=d.get('label'), markersize=markersz)
    
  x1 = ax.get_xlim()

  for d in data:
    mu = d['mu']
    ax.plot( x1, (mu,mu), color=d.get('color2'), lw=lw )
    print d.get('label') + ':' + str(mu) + ', ' + str(mu/offsets[5]*100)

  i = len(offsets)/2 + 1
  y = offsets[i]
  ax.plot( x1, (y,y), color='black', lw=lw )

  ax.set_xlabel('Sample')
  ax.set_ylabel('Absolute Error')
  title = 'Fit - ' + alg.upper()
  ax.set_title(title)
  #ax.autoscale(True, axis='y', tight=False)
  if legend:
    ax.legend(loc=1)

  ax.title.set_fontsize(20)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)
  ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
  if lims:
    ax.set_ylim(lims[1])
    ax.set_xlim(lims[0])

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
"""
Creates a new matplotlib figure for motor frequency analyses.  When renderering
motor frequency plots, this function should be used to initialize the figure
"""
def motorplot_figure():
  fig, ax = plt.subplots() 
  return fig, ax 

#------------------------------------------------------------------------------
"""
Plots the motor frequency with respect to time as a line plot.  For vicon, the 
motor frequency must be interpolated, so this plot may be misleading as it will
result in a step function when using a constant velocity motor model
"""
def motorplot_wrt_time(fig, ax, session, hz, title):
  x = session.get('t')
  y = session.get('hz')
  label = session.get('label')
  #title = 'Interpolated Actuator Frequency - ' + label 
  ax.plot(x, y, lw='1', label=label)
  xhz = ax.get_xlim()
  #ax.plot( xhz, (hz,hz), color='black', linewidth=1 )
  ax.set_title(title)
  ax.set_xlabel('Vicon Time (s)')
  ax.set_ylabel('Frequency (Hz)')

  ax.title.set_fontsize(16)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)

#------------------------------------------------------------------------------
"""
Plots the motor frequency with respect to samples as a line plot.  For vicon, 
the motor frequency must be interpolated, so this plot may be misleading as it 
will result in a step function when using a constant velocity motor model
"""
def motorplot_wrt_samples(fig, ax, session, hz, title):
  x = session.get('i')
  y = session.get('hz')
  label = session.get('label')
  #title = 'Interpolated Actuator Frequency - ' + label 
  ax.plot(x, y, lw='1', label=label)
  xhz = ax.get_xlim()
  #ax.plot( xhz, (hz,hz), color='black', linewidth=1 )
  ax.set_title(title)
  ax.set_xlabel('Sample')
  ax.set_ylabel('Frequency (Hz)')

  ax.title.set_fontsize(16)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)

#------------------------------------------------------------------------------
"""
"""
def motorplot_histogram(fig, ax, session, title, bins = 25):
  x = session.get('hz')
  label = session.get('label')
  #title = 'Actuator Frequency Histogram - ' + label 
  ax.hist(x, bins)
  ax.set_title(title)
  ax.set_xlabel('Frequency (Hz)')
  ax.set_ylabel('Samples')

  ax.title.set_fontsize(16)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)

  ax.set_xlim([1.5,3.5])

#------------------------------------------------------------------------------
"""
"""
def motorplot_scatter(fig, ax, session, hz, title, markersz = 2, ylim=[], color='r'):
  x = session.get('i')
  y = session.get('hz')
  label = session.get('label')
  #title = 'Actuator Frequency - ' + label 
  ax.scatter(x, y, marker='o', s=markersz, label=label, color=color)
  xi = ax.get_xlim()[0]
  xf = ax.get_xlim()[1]
  e = Ellipse(((xi + xf)/2,hz),xf-xi, 0.1)
  e.set_alpha(0.2)
  ax.add_artist(e)
  ax.set_title(title)
  ax.set_xlabel('Sample')
  ax.set_ylabel('Frequency (Hz)')

  ax.title.set_fontsize(16)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)
  #ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
  if ylim:
    ax.set_ylim(ylim)

#------------------------------------------------------------------------------
"""
"""
def motorplot_group_scatter(fig, ax, data, hz, title, markersz = 2, ellipseht=0.2, ylim=[]):
  for d in data:
    x = d.get('i')
    y = d.get('hz')
    label = d.get('label')
    ax.scatter(x, y, marker='o', s=markersz, label=label)

  xi = ax.get_xlim()[0]
  xf = ax.get_xlim()[1]
  e = Ellipse(((xi + xf)/2,hz),xf-xi, ellipseht)
  e.set_alpha(0.2)
  ax.add_artist(e)
  ax.set_title(title)
  ax.set_xlabel('Sample')
  ax.set_ylabel('Frequency (Hz)')

  box = ax.get_position()
  ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
  plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='large', markerscale=2)

  ax.title.set_fontsize(16)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)
  #ax.yaxis.get_major_formatter().set_powerlimits((0, 1))
  if ylim:
    ax.set_ylim(ylim)


#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
"""
Creates a new matplotlib figure for mocap analyses.  When renderering mocap 
plots, this function should be used to initialize the figure
"""
def mocapplot_figure():
  fig, ax = plt.subplots() 
  return fig, ax 

#------------------------------------------------------------------------------
"""
"""
def mocapplot_z(fig, ax, t, z, actual_mu, desired_mu):
  color_actual_plot = '#2080b0'
  color_actual_mu = '#0000ff'
  color_desired_mu = '#ff0000'
  #color_desired_plot = '#ff0000'
  #hilit_desired_mu = '#ffb0b0'
  
  ax.set_title('Measured Vertical Position')
  ax.set_xlabel('Virtual Time (s)')
  ax.set_ylabel(r'meters')
  ax.title.set_fontsize(20)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)
  ax.set_xlim([0,120])
  ax.set_ylim([0.03,0.09])
  ax.plot(t, z, color=color_actual_plot, lw=1)
  x2 = ax.get_xlim()
  ax.plot(x2, (actual_mu,actual_mu), color=color_actual_mu)
  ax.plot(x2, (desired_mu,desired_mu), color=color_desired_mu)

#------------------------------------------------------------------------------
"""
"""
def mocapplot_z_histogram(fig, ax, z, actual_mu, desired_mu):
  color_actual_plot = '#2080b0'
  color_actual_mu = '#0000ff'
  color_desired_mu = '#ff0000'
  #color_desired_plot = '#ff0000'
  #hilit_desired_mu = '#ffb0b0'
  
  ax.set_title('Vertical Position Distribution')
  ax.set_xlabel(r'meters')
  ax.set_ylabel(r'samples')
  ax.title.set_fontsize(20)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)
  ax.set_xlim([0.02,0.07])

  bins = 100
  ax.hist(z, bins, edgecolor=color_actual_plot, facecolor='#ffffff', lw=1, histtype='stepfilled')
  y2 = ax.get_ylim()
  ax.plot((actual_mu,actual_mu), y2, color=color_actual_mu)
  ax.plot((desired_mu,desired_mu), y2, color=color_desired_mu)

#------------------------------------------------------------------------------
"""
"""
def mocapplot_z_distribution_cache(fig, ax, z):
  ax.set_title('Figure Buffer')
  n, bins, patches = ax.hist(z, 200, histtype='stepfilled')
  close(fig) 
  return n, bins, patches

def mocapplot_z_distribution(fig, ax, z, actual_mu, desired_mu, n, bins, patches):
  color_actual_plot = '#2080b0'
  color_actual_mu = '#0000ff'
  color_desired_mu = '#ff0000'
  #color_desired_plot = '#ff0000'
  hilit_desired_mu = '#ffb0b0'
  color_epsilon = '#df7f3f'
  
  ax.set_title('Vertical Position Gaussian')
  ax.set_xlabel(r'meters')
  ax.set_yticklabels([])
  ax.set_yticks([])
  ax.title.set_fontsize(20)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.set_xlim([0.023,0.057])
  ax.set_ylim([0,120])

  sigma = z.std()
  y1 = mlab.normpdf(bins, actual_mu, sigma)
  y2 = ax.get_ylim()

  ax.plot((actual_mu,actual_mu), y2, color=color_actual_mu)
  ax.plot((desired_mu,desired_mu), y2, color=hilit_desired_mu,lw=5)
  ax.plot((desired_mu,desired_mu), y2, color=color_desired_mu)

  x_min = actual_mu - 3 * sigma
  x_max = actual_mu + 3 * sigma

  ax.plot((x_min,x_min), y2, color=color_epsilon)
  ax.plot((x_max,x_max), y2, color=color_epsilon)

  ax.plot(bins, y1, color=color_actual_plot)

#------------------------------------------------------------------------------
"""
"""
def mocapplot_dtheta(fig, ax, t, dtheta, actual_mu):
  color_actual_plot = '#2080b0'
  color_actual_mu = '#0000ff'
  #color_desired_mu = '#ff0000'
  #color_desired_plot = '#ff0000'
  #hilit_desired_mu = '#ffb0b0'
  
  ax.set_title('Computed Change in Orientation')
  ax.set_xlabel('Virtual Time (s)')
  ax.set_ylabel(r'radians')
  ax.title.set_fontsize(20)
  ax.xaxis.label.set_fontsize(16)
  for item in ax.get_xticklabels():
    item.set_fontsize(12)
  ax.yaxis.label.set_fontsize(16)
  for item in ax.get_yticklabels():
    item.set_fontsize(12)
  ax.set_xlim([0,120])
  ax.set_ylim([0,1.6])
  ax.plot(t, dtheta, color=color_actual_plot, lw=1)
  x2 = ax.get_xlim()
  ax.plot(x2, (actual_mu,actual_mu), color=color_actual_mu)

#------------------------------------------------------------------------------

