function [ q ] = quatrpy( r, p, y )
%QUATRPY Summary of this function goes here
%   Detailed explanation goes here

  q = zeros(1,4);

  phi = r * 0.5;
  the = p * 0.5;
  psi = y * 0.5;

  cphi = cos(phi);
  sphi = sin(phi);
  cpsi = cos(psi);
  spsi = sin(psi);
  cthe = cos(the);
  sthe = sin(the);

  q(1,4) = cphi * cthe * cpsi + sphi * sthe * spsi;
  q(1,1) = sphi * cthe * cpsi - cphi * sthe * spsi;
  q(1,2) = cphi * sthe * cpsi + sphi * cthe * spsi;
  q(1,3) = cphi * cthe * spsi - sphi * sthe * cpsi;
  
end

