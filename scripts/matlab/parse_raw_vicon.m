function [ x ] = parse_raw_vicon( txt1, txt2, txt3, txt4 )
%PARSE_RAW_VICON Summary of this function goes here
%   Detailed explanation goes here

  x_toks = regexp( txt2, '[:,]', 'split' );
  x1 = str2double( x_toks(2) );  % x
  x2 = str2double( x_toks(3) );  % y
  x3 = str2double( x_toks(4) );  % z
  
  q_toks = regexp( txt3, '[:,]', 'split' );
  q1 = str2double( q_toks(2) );  % x
  q2 = str2double( q_toks(3) );  % y
  q3 = str2double( q_toks(4) );  % z
  q4 = str2double( q_toks(5) );  % w
  
  x = [x1, x2, x3, q1, q2, q3, q4];
end

