import numpy as np

class quaternion:
  def __init__(self, q):
    self.x = q.x
    self.y = q.y
    self.z = q.z
    self.w = q.w
  def __init__(self, x=0, y=0, z=0, w=1):
    self.x = x
    self.y = y
    self.z = z
    self.w = w
  def deriv( q, w ):
    qd = quaternion()
    qd.w = 0.5 * ( -q.x * w[0] - q.y * w[1] - q.z * w[2] )
    qd.x = 0.5 * ( +q.w * w[0] + q.z * w[1] - q.y * w[2] )
    qd.y = 0.5 * ( -q.z * w[0] + q.w * w[1] + q.x * w[2] )
    qd.z = 0.5 * ( +q.y * w[0] - q.x * w[1] + q.w * w[2] )    
    return qd
  def to_omega( q, qd ):
    omega = [0,0,0]
    omega[0] = 2 * (-q.x * qd.w + q.w * qd.x - q.z * qd.y + q.y * qd.z)
    omega[1] = 2 * (-q.y * qd.w + q.z * qd.x + q.w * qd.y - q.x * qd.z)
    omega[2] = 2 * (-q.z * qd.w - q.y * qd.x + q.x * qd.y + q.w * qd.z)
    return omega
  def calc_angle( q1, q2 ):
    dot = q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w;
    if dot < -1.0:
      dot = -1.0
    elif dot > 1.0:
      dot = 1.0
    return acos(abs(dot));
  def normalize( self ):
    d = np.sqrt(self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w)
    self.x = self.x / d
    self.y = self.y / d
    self.z = self.z / d
    self.w = self.w / d
