function [ result ] = quatrot( q, v )
%QUATROT Summary of this function goes here
%   Detailed explanation goes here

  result = zeros(1,3);

  w2 = q(1,4)*q(1,4);
  x2 = q(1,1)*q(1,1);
  y2 = q(1,2)*q(1,2);
  z2 = q(1,3)*q(1,3);
  xy = q(1,1)*q(1,2);
  xz = q(1,1)*q(1,3);
  yz = q(1,2)*q(1,3);
  xw = q(1,1)*q(1,4);
  yw = q(1,2)*q(1,4);
  zw = q(1,3)*q(1,4);
  
  result(1,1) = (-1.0+2.0*(w2+x2))*v(1,1) + 2.0*((xy-zw)*v(1,2) + (yw+xz)*v(1,3));
  result(1,2) = 2.0*((xy+zw)*v(1,1) + (-xw+yz)*v(1,3)) + (-1.0+2.0*(w2+y2))*v(1,2); 
  result(1,3) = 2.0*((-yw+xz)*v(1,1) + (xw+yz)*v(1,2)) + (-1.0+2.0*(w2+z2))*v(1,3);
 
  % x = (-1.0+2.0*(w2+x2))*o.x() + 2.0*((xy-zw)*o.y() + (yw+xz)*o.z())
  % y = 2.0*((xy+zw)*o.x() + (-xw+yz)*o.z()) + (-1.0+2.0*(w2+y2))*o.y()
  % z = 2.0*((-yw+xz)*o.x() + (xw+yz)*o.y()) + (-1.0+2.0*(w2+z2))*o.z()
end

