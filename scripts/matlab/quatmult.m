function [ q ] = quatmult( q1, q2 )
%QUATMULT Summary of this function goes here
%   Detailed explanation goes here

  q = zeros(1,4);

  q(1,4) = q1(1,4) * q2(1,4) - q1(1,1) * q2(1,1) - q1(1,2) * q2(1,2) - q1(1,3) * q2(1,3);
  q(1,1) = q1(1,4) * q2(1,1) + q1(1,1) * q2(1,4) + q1(1,2) * q2(1,3) - q1(1,3) * q2(1,2);
  q(1,2) = q1(1,4) * q2(1,2) + q1(1,2) * q2(1,4) + q1(1,3) * q2(1,1) - q1(1,1) * q2(1,3);
  q(1,3) = q1(1,4) * q2(1,3) + q1(1,3) * q2(1,4) + q1(1,1) * q2(1,2) - q1(1,2) * q2(1,1);

end

