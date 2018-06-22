# run script in common parent directory of scripts and data directories
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import re
import math

#------------------------------------------------------------------------------
def round_to_hundreth( t ):
  return round(t,2)
  x = float(t) * float(100)
  i = int(x)
  d = x - float(int(x))
  if(d >= 0.5):
    i += 1
  return float(i) / float(100)

#------------------------------------------------------------------------------
def save_dataset(path, data):
  A = np.zeros((len(data),18))
  i = 0
  for s in data:
    x = s[3]
    q = s[4]
    dx = s[5]
    dq = s[6]
    m = s[7]
    A[i] = [s[0], s[1], s[2], x[0], x[1], x[2], q[0], q[1], q[2], q[3], dx[0], dx[1], dx[2], dq[0], dq[1], dq[2], m[0], m[1] ]
    i = i + 1
  #np.savetxt(path, A, delimiter=" ", fmt="%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f")
  np.savetxt(path, A, delimiter=" ", fmt="%d %d %f %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e")
#------------------------------------------------------------------------------
def save_testset(path, data):
  A = np.zeros((len(data),30))
  i = 0
  #print data
  for s in data:
#    x0 = s[3]
#    q0 = s[4]
#    dx0 = s[5]
#    dq0 = s[6]
#    m0 = s[7]
#    x1 = s[11]
#    q1 = s[12]
#    dx1 = s[13]
#    dq1 = s[14]
#    m1 = s[15]
#    A[i] = [x0[0], x0[1], x0[2], q0[0], q0[1], q0[2], q0[3], dx0[0], dx0[1], dx0[2], dq0[0], dq0[1], dq0[2], m0[0], m0[1], x1[0], x1[1], x1[2], q1[0], q1[1], q1[2], q1[3], dx1[0], dx1[1], dx1[2], dq1[0], dq1[1], dq1[2], m1[0], m1[1]]
    #print s
    A[i] = [s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], s[11], s[12], s[13], s[14], s[15], s[16], s[17], s[21], s[22], s[23], s[24], s[25], s[26], s[27], s[28], s[29], s[30], s[31], s[32], s[33], s[34], s[35]]
    i = i + 1
  np.savetxt(path, A, delimiter=" ", fmt="%1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e")
#------------------------------------------------------------------------------

class keyframe:
  frame = -1
  reflection = False
  twoframe = False
  sample = -1
  video_time = -1
  vicon_time = -1
#  virtual_time = -1

  def __init__(self, data = []):
    if(len(data)):
      self.parse(data)

  def parse(self, data):
    a = re.split( ' ', data )
    self.frame = int(a[0])
    if(len(a) > 1):
      if '+' in a[1]:
        self.twoframe = True 
      if '*' in a[1]:
        self.reflection = True 

  def echo(self):
    print(self.serialize())

  def serialize(self):
    s = "frame:" + str(self.frame)
    s += ", sample:" + str(self.sample)
    s += ", video_time:" + str(self.video_time)
    s += ", vicon_time:" + str(self.vicon_time)
#    s += ", virtual_time:" + str(self.virtual_time)
    s += ", reflection:" + str(self.reflection)
    s += ", twoframe:" + str(self.twoframe)
    return s

#------------------------------------------------------------------------------
class signal_data:
  synch_frame = -1
  synch_vicon_t = -1
  synch_virtual_t = -1
  video_frames = -1
  video_framerate = -1
  vicon_start_frame = -1
  vicon_samplerate = -1
  data = []
 
  def __init__(self, filepath):
    self.filepath = filepath

  def read(self):
    f = open( self.filepath, 'r' )
    content = [x.strip('\r\n') for x in f.readlines() ]

    #for line in content:
    #  print(line)

    #print(len(content))
    for i in range(len(content)):
      if content[i][0] == '#':
#        print(content[i])
        continue

      if i == 1:
        self.synch_frame = int(content[i])
      elif i == 3:
        self.synch_vicon_t = float(content[i])
      elif i == 5:
        self.synch_virtual_t = float(content[i])
      elif i == 7:
        self.video_frames = int(content[i])
      elif i == 9:
        self.video_framerate = int(content[i])
      elif i == 11:
        self.vicon_start_frame = int(content[i])
      elif i == 13:
        self.vicon_samplerate = int(content[i])
      elif i >= 15:
        if(i == 15):
          kfs = []
        line = content[i]
        if(len(line) == 1 and line == '-'):
          if(len(kfs)>0):
            self.data.append(kfs)
            kfs = []
          #print('Gap')
          continue
	else:
          kf = keyframe(line)
          #kf.echo()
          kfs.append(kf)
#        print(content[i])
#        continue
#      print(content[i])


    f.close()

  def echo(self, verbose = True):
    print( "synch_frame: " + str(self.synch_frame) )
    print( "synch_vicon_t: " + str(self.synch_vicon_t) )
    print( "synch_virtual_t: " + str(self.synch_virtual_t) )
    print( "video_frames: " + str(self.video_frames) )
    print( "video_framerate: " + str(self.video_framerate) )
    print( "vicon_start_frame: " + str(self.vicon_start_frame) )
    print( "vicon_samplerate: " + str(self.vicon_samplerate) )
    if(verbose):
      for kfs in self.data:
        s = "[" + str(len(kfs))
        first = True
        for kf in kfs:
          if not first:
            s += ", " 
          s += "[" + kf.serialize() + "]"
          first = False
        s += "]"
        print(s)

  def frames2time(self, frames):
    return float(frames)/float(self.video_framerate)
  def time2frames(self, time):
    return int(float(time)*float(self.video_framerate))
    #return float(time)*float(self.video_framerate)
  def sample2time(self, sample):
    return float(sample)/float(self.vicon_samplerate)
  def time2sample(self, time):
    return int(round(time*float(self.vicon_samplerate)))

#------------------------------------------------------------------------------
class vicon_data:
  data = []
  filepath = ""

  def __init__(self, filepath = "" ):
    self.filepath = filepath
    if filepath == "":
      self.data == np.ndarray(shape=(),dtype='int32, int32, float64, 3float64, 4float64, 3float64, 3float64, 2float64')

  def read(self):
    if self.filepath != "":
      # video-frame-idx, vicon-sample-idx, vicon-time, pos(x,y,z), rot(x,y,z,w), linvel(x,y,z), angvel(x,y,z)    
      self.data =  np.loadtxt( self.filepath, usecols=(range(18)), dtype='int32, int32, float64, 3float64, 4float64, 3float64, 3float64, 2float64' )
      print("shape:"+str(self.data.shape))

  def write(self, path):
    save_dataset(path, self.data)
#    self.echo()
#    A = np.zeros((self.samples(),18))
#    i = 0
#    for s in self.data:
#      x = s[3]
#      q = s[4]
#      dx = s[5]
#      dq = s[6]
#      m = s[7]
#      A[i] = [s[0], s[1], s[2], x[0], x[1], x[2], q[0], q[1], q[2], q[3], dx[0], dx[1], dx[2], dq[0], dq[1], dq[2], m[0], m[1] ]
#      i = i + 1
#    #np.savetxt(path, A, delimiter=" ", fmt="%d %d %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f")
#    np.savetxt(path, A, delimiter=" ", fmt="%d %d %f %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e %1.8e")

  def echo(self, sample_idx = -1):
    if sample_idx < 0:
      s = "["
      s += "samples:" + str(self.samples())
      s += ", fields:" + str(self.fields())
      s += "]"
      print( s )
      print( self.data )
    else:
      print(self.data[sample_idx])

  def samples( self ):
    return len(self.data)
  def fields( self ):
    return len(self.data[0])

  def get_frame_idx( self, sample_idx ):
    return self.data[sample_idx][0]
  def set_frame_idx( self, sample_idx, frame_idx ):
    self.data[sample_idx][0] = frame_idx

  def get_vicon_time( self, sample_idx ):
    return self.data[sample_idx][2]
  def set_vicon_time( self, sample_idx, t ):
    self.data[sample_idx][2] = t

  def get_vicon_position( self, sample_idx ):
    return self.data[sample_idx][3]
  def set_vicon_position( self, sample_idx, x ):
    self.data[sample_idx][3] = x

  def get_vicon_rotation( self, sample_idx ):
    return self.data[sample_idx][4]
  def set_vicon_rotation( self, sample_idx, q ):
    self.data[sample_idx][4] = q

  def get_vicon_linvel( self, sample_idx ):
    return self.data[sample_idx][5]
  def set_vicon_linvel( self, sample_idx, dx ):
    self.data[sample_idx][5] = dx

  def get_vicon_angvel( self, sample_idx ):
    return self.data[sample_idx][6]
  def set_vicon_angvel( self, sample_idx, omega ):
    self.data[sample_idx][6] = omega
  
#==============================================================================
## TODO parameterize through args
trialid = 10

## IMPORTANT NOTE: the data used in development was uncorrected.  Center 
## correction and interpolation was not performed on this data.  It does not 
## affect this algorithm; however, it will affect vicon data structure.
signal_data_path = 'data/vicon/interpreted/signals/'
vicon_data_path = 'data/vicon/phase1_corrected/'  
#output_path = 'data/vicon_mapped/'
output_path = 'data/vicon/'

## build the standardized filename
rootname = 'trial_' + str(trialid).zfill(2)
filename = rootname + '.txt'

## read in all the signal data for this trial
sigs = signal_data( signal_data_path + filename )
sigs.read()

## compute the video time for all keyframes
for kfs in sigs.data:
  for kf in kfs:
    kf.video_time = sigs.frames2time(kf.frame)

## print the first set of keyframes for examination
#kfs = sigs.data[0]
#for kf in kfs:
#  kf.echo()

## read in all the vicon data for this trial
vicon = vicon_data( vicon_data_path + filename )
vicon.read()

## find the vicon sample of the synchronization collision and create a keyframe
skf = keyframe()
skf.frame = sigs.synch_frame
for i in range(vicon.samples()):
  vicon_t = vicon.get_vicon_time(i)
  if(vicon_t == sigs.synch_vicon_t):
    skf.sample = i
    break
  if(vicon_t > sigs.synch_vicon_t):
    t1 = vicon_t
    t0 = vicon.get_vicon_time(i-1)
    delta = t1 - t0;
    #print("t0:"+str(t0) + ", t1:"+str(t1) + ", delta:"+str(delta))
    if(sigs.synch_vicon_t < t0 + delta/2):
      skf.sample = i-1
    else:
      skf.sample = i
    diff = abs(sigs.synch_vicon_t - vicon.get_vicon_time(skf.sample))
    print("WARNING: found nearest synchronization sample[diff:" + str(diff)+"]")
    break

## if a synchronization sample could not be found, there is bad juju
if skf.sample == -1:
  print("ERROR: unable to find a synchronization sample in vicon data")
  exit(1)

skf.video_time = sigs.frames2time(sigs.synch_frame)
skf.vicon_time = sigs.synch_vicon_t
#print("skf.sample::" + skf.serialize())
#sigs.echo(False)
#vicon.echo(0)
#vicon.echo(skf.sample)

## will throw out all keyframes that occurred before vicon t0
vicon0_kf = keyframe()
vicon0_kf.vicon_time = 0
#vicon0_kf.virtual_time = 0
vicon0_kf.sample = 0
vicon0_kf.frame = skf.frame - sigs.time2frames(skf.video_time - skf.vicon_time)
vicon0_kf.video_time = skf.video_time - skf.vicon_time
#print("vicon0_kf::" + vicon0_kf.serialize())

## compute the vicon time for all keyframes
for kfs in sigs.data:
  for kf in kfs:
    kf.vicon_time = kf.video_time - vicon0_kf.video_time
    kf.sample = sigs.time2sample(kf.vicon_time)

## print the first set of keyframes for examination
#kfs = sigs.data[0]
#for kf in kfs:
#  kf.echo()

## interpolate rotation for each sample that falls between keyframes
## where the keyframe represents the zero position of the joint
#samples = vicon_data()
for kfs in sigs.data:
  # if this set of keyframes has only one, can't interpolate
  if len(kfs) == 1:
    continue
  for i in range(len(kfs)-1):
    kf0 = kfs[i]           # beginning keyframe
    kf1 = kfs[i+1]         # end keyframe 
    s0 = vicon.data[kf0.sample]
    s1 = vicon.data[kf1.sample]
    delta_s = kf1.sample - kf0.sample
    delta_t = float(kf1.sample - kf0.sample) / float(100)
    #delta_t = (kf1.video_time - kf0.video_time)
    #print("delta_t:" + str(delta_t) + ", " + str(kf1.sample) + ", " + str(kf0.sample))
    theta = 0
    dtheta = 2*math.pi/delta_t
    for j in range(delta_s):
      #s = vicon.data[kf0.sample+j]
      #print("theta,dtheta:" + str(theta) + "," + str(dtheta))
      vicon.data[kf0.sample+j][7][0] = theta
      vicon.data[kf0.sample+j][7][1] = dtheta
      theta += dtheta * 0.01
      #np.append(arr=samples.data, values=vicon.data[kf0.sample+j])

fig,ax = plt.subplots()
#sim dtheta
x = []
y = []
for s in vicon.data:
  x.append(s[2])
  y.append(2.5 * 2 * math.pi)
hdl, = ax.plot(x, y, color='#000000')
#most prominant dtheta
#x = []
#y = []
#for s in vicon.data:
#  x.append(s[2])
#  y.append(u[j])
#hdl, = ax.plot(x, y, color='#00ff00')
#theta
x = []
y = []
for s in vicon.data:
  x.append(s[2])
  y.append(s[7][0])
hdl, = ax.plot(x, y, color='#0000ff')
#dtheta
x = []
y = []
for s in vicon.data:
  x.append(s[2])
  y.append(s[7][1])
hdl, = ax.plot(x, y, color='#ff0000')
u, idxs, cts = np.unique(y,return_index=True,return_counts=True)
#print(u)
#print(cts)
m = 0
j = 0
for i in range(len(cts)):
  c = cts[i]
  if c > 0:
    if u[i] > 0.10:
      if c > m:
        m = c
        j = i
#print(cts[j])

#plt.show()
imgname = rootname + '.png'
plt.savefig(imgname, dpi=200, transparent=True)
vicon.write(rootname+'.txt')

freq = u[j]/(2*math.pi)
#print("freq:"+str(freq))

#NOTE: This process needs to save records that are state pairs (t0, t1) rather than individual records
A = np.empty((0), dtype='int32, int32, float64, 3float64, 4float64, 3float64, 3float64, 2float64')
for s in vicon.data:
  if s[7][1] == u[j]:
    A = np.append(A, s)
fname = rootname + '_2_5Hz.txt'
save_dataset(fname, A)

#B = np.empty((0), dtype='int32, int32, float64, 3float64, 4float64, 3float64, 3float64, 2float64, int32, int32, float64, 3float64, 4float64, 3float64, 3float64, 2float64')
B = np.empty((0))
for i in range(len(A)-1):
  x0 = A[i]
  x1 = A[i+1]
  s0 = x0[1]
  s1 = x1[1]
  if s0+1 != s1:
    continue
  #x = [x0[0], x0[1], x0[2], x0[3], x0[4], x0[5], x0[6], x0[7], x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], x1[6], x1[7]]
  x = [x0[0], x0[1], x0[2], x0[3][0], x0[3][1], x0[3][2], x0[4][0], x0[4][1], x0[4][2], x0[4][3], x0[5][0], x0[5][1], x0[5][2], x0[6][0], x0[6][1], x0[6][2], x0[7][0], x0[7][1], x1[0], x1[1], x1[2], x1[3][0], x1[3][1], x1[3][2], x1[4][0], x1[4][1], x1[4][2], x1[4][3], x1[5][0], x1[5][1], x1[5][2], x1[6][0], x1[6][1], x1[6][2], x1[7][0], x1[7][1]]
  #x = [x0, x1]
  #x = (x0, x1)
  #x = np.concatenate((x0,x1))
  #print x
  B = np.append(B, x)
  #B = np.append(B, (x0, x1))
#rows=len(B)/30
B=np.reshape(B, (len(B)/36,36))
#print B
fname = rootname + '_testset.txt'
save_testset(fname, B)

#t:1, freq:2.5
#t:2, freq:2.5
#t:3, freq:2.5
#t:4, freq:2.5
#t:5, freq:2.5
#t:6, freq:2.326
#t:7, freq:2.5
#t:8, freq:2.5
#t:9, freq:2.5
#t:10, freq:2.5

