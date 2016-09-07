function [ epsilon ] = centeroffset_objectivefun( u )
%CENTEROFFSET_OBJECTIVEFUN Summary of this function goes here
%   Detailed explanation goes here

  load('V.mat');  % loads the vicon data into variable q

  r = 0.041;
  epsilon = 0;
  Rm2v = quatrpy(0,0,pi);
  Rm2v = Rm2v / norm(Rm2v);
  
  for idx = 1:size(V,1)
    session = V(idx,1);
    frame_id = V(idx, 2);
    q = V(idx,3:9);
    if( ~is_good_frame( session, frame_id ) )
        continue;
    end
    
    qvxyz = q(1,1:3);
    Rqv = q(1,4:7);
    Rqv = Rqv / norm(Rqv);
    R = quatmult(Rm2v, Rqv);
    R = R / norm(R);
    
    uv = quatrot( R, u );
    
    cv = qvxyz + uv;
    delta = (cv(1,3) - r)^2;
    
    epsilon = epsilon + delta;
  end
  
end

