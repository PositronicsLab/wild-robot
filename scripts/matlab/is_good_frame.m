function [ result ] = is_good_frame( session, frame_idx )
%IS_GOOD_FRAME Summary of this function goes here
%   Detailed explanation goes here

  % frame_idx is zero based
  
  result = 1;
  if( session == 1 ) 
    % session  1: 352,354; 1733,1734; 11911,11912
    if( frame_idx < 54 )
      result = 0;
    elseif( frame_idx >= 352 && frame_idx <= 354 ) 
      result = 0;
    elseif( frame_idx >= 1733 && frame_idx <= 1734 ) 
      result = 0;
    elseif( frame_idx >= 11911 && frame_idx <= 11912 ) 
      result = 0;
    end
  elseif( session == 2 )
    % session  2: 1367,1368; 10133,10134; 12153,12154
    if( frame_idx < 36 )
      result = 0;
    elseif( frame_idx >= 1367 && frame_idx <= 1368 ) 
      result = 0;
    elseif( frame_idx >= 10133 && frame_idx <= 10134 ) 
      result = 0;
    elseif( frame_idx >= 12153 && frame_idx <= 12154 ) 
      result = 0;
    end
  elseif( session == 3 ) 
    % session  3: 14,15; 477,481; 3046,3068; 3404,3406; 5981,5982; 7843,7845; 9600,9601
    if( frame_idx < 100 )
      result = 0;
    elseif( frame_idx >= 14 && frame_idx <= 15 ) 
      result = 0;
    elseif( frame_idx >= 477 && frame_idx <= 481 ) 
      result = 0;
    elseif( frame_idx >= 3046 && frame_idx <= 3068 ) 
      result = 0;
    elseif( frame_idx >= 3404 && frame_idx <= 3406 ) 
      result = 0;
    elseif( frame_idx >= 5981 && frame_idx <= 5982 ) 
      result = 0;
    elseif( frame_idx >= 7843 && frame_idx <= 7845 ) 
      result = 0;
    elseif( frame_idx >= 9600 && frame_idx <= 9601 ) 
      result = 0;
    end
  elseif( session == 4 ) 
    % session  4: 984,986; 2699,2700; 6791,6792; 6975,6976
    if( frame_idx < 47 )
      result = 0;
    elseif( frame_idx >= 984 && frame_idx <= 986 ) 
      result = 0;
    elseif( frame_idx >= 2699 && frame_idx <= 2700 ) 
      result = 0;
    elseif( frame_idx >= 6791 && frame_idx <= 6792 ) 
      result = 0;
    elseif( frame_idx >= 6975 && frame_idx <= 6976 ) 
      result = 0;
    end
  elseif( session == 5 ) 
    % session  5: 20,22; 33,35; 11216,11218; 11227,11228
    if( frame_idx < 25 )
      result = 0;
    elseif( frame_idx >= 20 && frame_idx <= 22 ) 
      result = 0;
    elseif( frame_idx >= 33 && frame_idx <= 35 ) 
      result = 0;
    elseif( frame_idx >= 11216 && frame_idx <= 11218 ) 
      result = 0;
    elseif( frame_idx >= 11227 && frame_idx <= 11228 ) 
      result = 0;
    end
  elseif( session == 6 ) 
    % session  6: 2381,2383; 2425,2427; 5573,5576; 9104,9105
    if( frame_idx < 66 )
      result = 0;
    elseif( frame_idx >= 2381 && frame_idx <= 2383 ) 
      result = 0;
    elseif( frame_idx >= 2425 && frame_idx <= 2427 ) 
      result = 0;
    elseif( frame_idx >= 5573 && frame_idx <= 5576 ) 
      result = 0;
    elseif( frame_idx >= 9104 && frame_idx <= 9105 ) 
      result = 0;
    end
  elseif( session == 7 ) 
    % session  7: 0,7; 16,19; 1103,1105; 1179,1180; 4423,4425; 4489,4491; 4855,4856; 4987,4988; 9757,9758
    if( frame_idx < 45 )
      result = 0;
    elseif( frame_idx >= 0 && frame_idx <= 7 ) 
      result = 0;
    elseif( frame_idx >= 16 && frame_idx <= 19 ) 
      result = 0;
    elseif( frame_idx >= 1103 && frame_idx <= 1105 ) 
      result = 0;
    elseif( frame_idx >= 1179 && frame_idx <= 1180 ) 
      result = 0;
    elseif( frame_idx >= 4423 && frame_idx <= 4425 ) 
      result = 0;
    elseif( frame_idx >= 4489 && frame_idx <= 4491 ) 
      result = 0;
    elseif( frame_idx >= 4855 && frame_idx <= 4856 ) 
      result = 0;
    elseif( frame_idx >= 4987 && frame_idx <= 4988 ) 
      result = 0;
    elseif( frame_idx >= 9757 && frame_idx <= 9758 ) 
      result = 0;
    end
  elseif( session == 8 )
    % session  8: 5362,5363; 8124,8130; 9561,9562
    if( frame_idx < 15 )
      result = 0;
    elseif( frame_idx >= 5362 && frame_idx <= 5363 ) 
      result = 0;
    elseif( frame_idx >= 8124 && frame_idx <= 8130 ) 
      result = 0;
    elseif( frame_idx >= 9561 && frame_idx <= 9562 ) 
      result = 0;
    end
  elseif( session == 9 ) 
    % session  9: 2638,2642; 3774,3775; 4280,4282; 5681,5682; 6435,6438; 9476,9478; 10869,10871; 10958,10959; 12167,12169
    if( frame_idx < 37 )
      result = 0;
    elseif( frame_idx >= 2638 && frame_idx <= 2642 ) 
      result = 0;
    elseif( frame_idx >= 3774 && frame_idx <= 3775 ) 
      result = 0;
    elseif( frame_idx >= 4280 && frame_idx <= 4282 ) 
      result = 0;
    elseif( frame_idx >= 5681 && frame_idx <= 5682 ) 
      result = 0;
    elseif( frame_idx >= 6435 && frame_idx <= 6438 ) 
      result = 0;
    elseif( frame_idx >= 9476 && frame_idx <= 9478 ) 
      result = 0;
    elseif( frame_idx >= 10869 && frame_idx <= 10871 ) 
      result = 0;
    elseif( frame_idx >= 10958 && frame_idx <= 10959 ) 
      result = 0;
    elseif( frame_idx >= 12167 && frame_idx <= 12169 ) 
      result = 0;
    end
  elseif( session == 10 )
    % session 10: 232,233; 2714,2715; 7168,7169; 8184,8185; 11940,11942
    if( frame_idx < 32 )
      result = 0;
    elseif( frame_idx >= 232 && frame_idx <= 233 ) 
      result = 0;
    elseif( frame_idx >= 2714 && frame_idx <= 2715 ) 
      result = 0;
    elseif( frame_idx >= 7168 && frame_idx <= 7169 ) 
      result = 0;
    elseif( frame_idx >= 8184 && frame_idx <= 8185 ) 
      result = 0;
    elseif( frame_idx >= 11940 && frame_idx <= 11942 ) 
      result = 0;
    end
  end
end

