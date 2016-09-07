function validateframedata( kfstats, kfevents, vicon_collision, vicon_fixedstep, video, keyframes, ids )
%VALIDATEFRAMEDATA Summary of this function goes here
%   Detailed explanation goes here

  assert( kfstats(1,1) == vicon_collision );
  assert( kfstats(1,2) == vicon_fixedstep );
  assert( kfstats(1,3) == video );

  [rows1, cols1] = size(keyframes);
  [rows2, cols2] = size(ids);
  [rows3, cols3] = size(kfevents);
  assert( rows1 == rows2 );
  assert( cols1 == cols2 && cols1 == rows3 );

  for col = 1:cols1
    sec = floor(keyframes(1,col));
    frame = (keyframes(1,col) - sec) * 100;
  
    assert( sec == kfevents(col, 1) );
    assert( cast(frame, 'int32') == kfevents(col, 2) );
    assert( ids(1,col) == kfevents(col, 3) );
  end
end

