function [ theta ] = quatangle( q1, q2 )
%QUATANGLE Summary of this function goes here
%   Detailed explanation goes here

  dot = q1(1,1)*q2(1,1) + q1(1,2)*q2(1,2) + q1(1,3)*q2(1,3) + q1(1,4)*q2(1,4);

  if (dot < -1.0)
    dot = -1.0;
  elseif (dot > 1.0)
    dot = 1.0;
  end

  theta = acos(abs(dot));
end

