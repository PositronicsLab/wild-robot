
filename = 'keyframe_01';

[kfstats, kfevents] = readkeyframefile( filename );

fps = kfstats(1,4);
framerate = 1/fps;

%normalize video time (s + framenumber) to compatible double time
[rows, cols] = size( kfevents );
video_events = [];
for row = 1:rows
  sec = cast( kfevents( row, 1 ), 'double' );
  frame = cast( kfevents( row, 2 ), 'double' );
  idx = cast( kfevents( row, 3 ), 'double' );
  
  video_t = sec + frame * framerate;
  video_events = [video_events; [video_t, idx]];
end
%video_events now contains the map between approximate time (still in
%camera temporal frame of referece) and the signal events

%compute the error between vicon timestamps and video timestamps
vicon_ti = kfstats(1, 1);
video_ti = kfstats(1, 3);
% the approximate differential between the video timestamp and the vicon
% timestamp
diff_ti = video_ti - vicon_ti;