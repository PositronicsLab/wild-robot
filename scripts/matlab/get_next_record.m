function [ x ] = get_next_record( fd )
%READ_RECORD Summary of this function goes here
%   Detailed explanation goes here

  txt1 = fgetl( fd );
  txt2 = fgetl( fd );
  txt3 = fgetl( fd );
  txt4 = fgetl( fd );

  x = parse_raw_vicon( txt1, txt2, txt3, txt4 );
end

