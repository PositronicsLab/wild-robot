"""
James Taylor

This example demonstrates plotting motor frequency data from processed vicon 
data which must contain interpolated motor frequency 
"""
import numpy as np
import wbplot as wbplt
import math

#------------------------------------------------------------------------------
"""
Read vicon data from disk for a given session and converts the motor velocity
from radians per second to frequency
"""
def read_vicon_data(session):
  fname = 'trial_' + str(session).zfill(2) + '.txt'
  d = np.loadtxt( fname )
  d[:,0] = np.arange(d.shape[0])
  d[:,17] = d[:,17] / (2.0 * math.pi)
  return d

#------------------------------------------------------------------------------
"""
Cull samples from motor velocity that are equal to zero or are very significant
outliers
"""
def cull_motor_velocity(data):
  mvel = data[data[:,17] > 0]
  mvel[:,0] = np.arange(mvel.shape[0])
  mvel = mvel[mvel[:,17] < 5]
  return mvel

#------------------------------------------------------------------------------
"""
"""
def read_session_and_process_into_dicts(session):
  f = read_vicon_data(session)
  c = cull_motor_velocity(f)
  label = 'Session ' + str(session)
  fdict = {'i':f[:,0], 't':f[:,2], 'hz':f[:,17], 'label':label, 'id':session}
  cdict = {'i':c[:,0], 'hz':c[:,17], 'label':label, 'id':session} 
  return {'file':fdict, 'culled':cdict}

#------------------------------------------------------------------------------
"""
v# variables contain raw vicon data
t# variables contain adjusted vicon data
"""
selectedhz = 2.5
show = False

data = []
for i in range(10):
  d = read_session_and_process_into_dicts(i+1)
  data.append(d)

fig, ax = wbplt.motorplot_figure()
session = data[0].get('file')
title = 'Interpolated Actuator Frequency - ' + session.get('label')
wbplt.motorplot_wrt_time(fig, ax, session, selectedhz, title)
imgname = 'vicon_frequency-session_' + str(session.get('id')).zfill(2) + '-raw.png'
if show:
  wbplt.show()
else:
  wbplt.save(imgname)
  wbplt.close(fig)

fig, ax = wbplt.motorplot_figure()
session = data[0].get('culled')
title = 'Culled Actuator Frequency - ' + session.get('label')
wbplt.motorplot_wrt_samples(fig, ax, session, selectedhz, title)
imgname = 'vicon_frequency-session_' + str(session.get('id')).zfill(2) + '-culled.png'
if show:
  wbplt.show()
else:
  wbplt.save(imgname)
  wbplt.close(fig)

fig, ax = wbplt.motorplot_figure()
session = data[0].get('culled')
title = 'Actuator Frequency Histogram - ' + session.get('label')
wbplt.motorplot_histogram(fig, ax, session, title)
imgname = 'vicon_frequency-session_' + str(session.get('id')).zfill(2) + '-histogram.png'
if show:
  wbplt.show()
else:
  wbplt.save(imgname)
  wbplt.close(fig)

ylim=[1.5,3.5]

fig, ax = wbplt.motorplot_figure()
session = data[0].get('culled')
title = 'Actuator Frequency Summary - ' + session.get('label')
wbplt.motorplot_scatter(fig, ax, session, selectedhz, title, ylim=ylim)
imgname = 'vicon_frequency-session_' + str(session.get('id')).zfill(2) + '-scatter.png'
if show:
  wbplt.show()
else:
  wbplt.save(imgname)
  wbplt.close(fig)


fig, ax = wbplt.motorplot_figure()
sessions = []
for i in range(6):
  sessions.append(data[i].get('culled'))
title = 'Frequency Before Battery Change'
wbplt.motorplot_group_scatter(fig, ax, sessions, selectedhz, title, ellipseht=0.1, ylim=ylim)
imgname = 'vicon_frequency-before_batterychg-scatter.png'
if show:
  wbplt.show()
else:
  wbplt.save(imgname)
  wbplt.close(fig)


fig, ax = wbplt.motorplot_figure()
sessions = []
for i in range(4):
  sessions.append(data[i+6].get('culled'))
title = 'Frequency After Battery Change'
wbplt.motorplot_group_scatter(fig, ax, sessions, selectedhz, title, ellipseht=0.1, ylim=ylim)
imgname = 'vicon_frequency-after_batterychg-scatter.png'
if show:
  wbplt.show()
else:
  wbplt.save(imgname)
  wbplt.close(fig)


