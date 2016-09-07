
[kfstats01, kfevents01] = readkeyframefile( 'keyframe_01' );

keyframe_t1 = [1.29, 2.13, 2.24, 3.08, 3.22, 4.05, 4.17, 5.0, 5.26, 6.11, 8.26, 8.28, 9.26, 10.09, ...
            10.21, 11.06, 11.20, 12.05, 12.16, 13.0, 13.12, 13.28, 14.14, 14.27, 15.09, ...
            15.22, 16.04, 16.17, 17.02];
id_t1 = [1,2,3,4,5,6,7,8,10,11,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35];
vicon_collision_ti_t1 = 1.33;                % this is from the system timestamp
vicon_collision_fixedstep_ti_t1 = 1.33;
% this is assuming vicon is at fixed framerate
video_collision_ti_t1 = 3.0;

validateframedata(kfstats01, kfevents01, vicon_collision_ti_t1, vicon_collision_fixedstep_ti_t1, video_collision_ti_t1, keyframe_t1, id_t1);
%assert( kfstats01(1,1) == vicon_collision_ti_t1 );
%assert( kfstats01(1,2) == vicon_collision_fixedstep_ti_t1 );
%assert( kfstats01(1,3) == video_collision_ti_t1 );

%[rows1, cols1] = size(keyframe_t1);
%[rows2, cols2] = size(id_t1);
%[rows3, cols3] = size(kfevents01);
%assert( rows1 == rows2 );
%assert( cols1 == cols2 && cols1 == rows3 );

%for col = 1:cols1
%  sec = floor(keyframe_t1(1,col));
%  frame = (keyframe_t1(1,col) - sec) * 100;
  
%  assert( sec == kfevents01(col, 1) );
%  assert( cast(frame, 'int32') == kfevents01(col, 2) );
%  assert( id_t1(1,col) == kfevents01(col, 3) );
%end

%trial 2
[kfstats02, kfevents02] = readkeyframefile( 'keyframe_02' );
keyframe_t2 = [1.10, 1.25, 2.06, 2.19, 3.01, 3.17, 4.00, 5.25, 6.11, 6.24, 7.05, 7.18, 8.00, 8.13, 9.14, 10.10 ];
id_t2 = [1,2,3,4,5,6,7,11,12,13,14,15,16,17,20,22];
% lost 8, 9, 10, 11, 18, 19?
vicon_collision_ti_t2 = 1.869;
vicon_collision_fixedstep_ti_t2 = 1.87;         % this is assuming vicon is at fixed framerate
video_collision_ti_t2 = 2.27;
validateframedata(kfstats02, kfevents02, vicon_collision_ti_t2, vicon_collision_fixedstep_ti_t2, video_collision_ti_t2, keyframe_t2, id_t2);

%trial 3
[kfstats03, kfevents03] = readkeyframefile( 'keyframe_03' );
keyframe_t3 = [3.07, 3.23, 4.05, 4.17, 4.29, 5.13, 5.29, 6.24, 7.05, 7.20, 8.05, 8.21, 9.05, 9.18, 10.01, 10.28, 11.14, 12.09 ];
id_t3 = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,18,19,21];
% lost 8, 17, 20

vicon_collision_ti_t3 = 2.062;    
vicon_collision_fixedstep_ti_t3 = 1.88;         % this is assuming vicon is at fixed framerate
video_collision_ti_t3 = 5.0;
validateframedata(kfstats03, kfevents03, vicon_collision_ti_t3, vicon_collision_fixedstep_ti_t3, video_collision_ti_t3, keyframe_t3, id_t3);

%trial 4
[kfstats04, kfevents04] = readkeyframefile( 'keyframe_04' );
keyframe_t4 = [ 1.17, 2.02, 2.13, 2.27, 3.11, 5.18, 6.01, 6.15, 7.16, 7.28, 8.11, 8.23, 9.21, 10.03, 10.20, 11.03, 11.16, 12.02 ];
id_t4 = [1,2,3,4,5,10,11,13,14,15,16,18,19,20,22,23,24,25];
% lost 6,7,8,9,10?,12,17,21
vicon_collision_ti_t4 = 1.481;
vicon_collision_fixedstep_ti_t4 = 1.48;         % this is assuming vicon is at fixed framerate
video_collision_ti_t4 = 3.1;
validateframedata(kfstats04, kfevents04, vicon_collision_ti_t4, vicon_collision_fixedstep_ti_t4, video_collision_ti_t4, keyframe_t4, id_t4);

%trial 5
[kfstats05, kfevents05] = readkeyframefile( 'keyframe_05' );
keyframe_t5 = [1.02, 1.14, 1.27, 2.09, 2.21, 3.09, 4.18, 5.02, 5.15, 5.28, 6.11, 6.28, 7.10, 7.23, 8.05, 8.17, 9.17, 10.12, 10.24  ];
id_t5 = [1,2,3,4,5,6,10,11,12,13,14,15,16,17,18,19,21,23,24 ];
% lost 7,8,9, 20, 22
vicon_collision_ti_t5 = 1.656;
vicon_collision_fixedstep_ti_t5 = 1.66;         % this is assuming vicon is at fixed framerate
video_collision_ti_t5 = 2.8;
validateframedata(kfstats05, kfevents05, vicon_collision_ti_t5, vicon_collision_fixedstep_ti_t5, video_collision_ti_t5, keyframe_t5, id_t5);

%trial 6
[kfstats06, kfevents06] = readkeyframefile( 'keyframe_06' );
keyframe_t6 = [2.03, 2.17, 2.29, 3.11, 3.23, 4.07, 4.22, 5.19, 6.01, 6.14, 7.00, 7.13, 8.00, 8.15, 8.28, 9.11, 9.23, 10.22, 11.07 ];
id_t6 = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21 ];
% lost 8,19
vicon_collision_ti_t6 = 2.219;
vicon_collision_fixedstep_ti_t6 = 2.22;         % this is assuming vicon is at fixed framerate
video_collision_ti_t6 = 4.3;
validateframedata(kfstats06, kfevents06, vicon_collision_ti_t6, vicon_collision_fixedstep_ti_t6, video_collision_ti_t6, keyframe_t6, id_t6);

%trial 7
%Note: New battery for motor inserted at this trial
[kfstats07, kfevents07] = readkeyframefile( 'keyframe_07' );
keyframe_t7 = [1.13, 1.25, 2.07, 2.21, 3.01, 3.13, 3.25, 4.06, 4.18, 4.29, 5.12, 7.21, 8.04, 8.14, 8.26, 9.05, 9.21, 10.03, 10.15];
id_t7 = [1,2,3,4,5,6,7,8,9,10,11,17,18,19,20,21,23,24,25];
% lost 12,13,14,15,16,17?,22?
vicon_collision_ti_t7 = 2.231;
vicon_collision_fixedstep_ti_t7 = 1.89;         % this is assuming vicon is at fixed framerate
video_collision_ti_t7 = 3.5;
validateframedata(kfstats07, kfevents07, vicon_collision_ti_t7, vicon_collision_fixedstep_ti_t7, video_collision_ti_t7, keyframe_t7, id_t7);

%trial 8
[kfstats08, kfevents08] = readkeyframefile( 'keyframe_08' );
keyframe_t8 = [1.08, 1.18, 2.00, 2.13, 2.28, 3.09, 3.21, 4.02, 4.27, 5.10, 5.21, 6.02, 6.14, 6.28, 7.10, 10.18, 10.28 ];
id_t8 = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,24,25 ];
% lost 9,17,18,19,20,21,22,23?
vicon_collision_ti_t8 = 2.105;
vicon_collision_fixedstep_ti_t8 = 2.11;         % this is assuming vicon is at fixed framerate
video_collision_ti_t8 = 2.4;
validateframedata(kfstats08, kfevents08, vicon_collision_ti_t8, vicon_collision_fixedstep_ti_t8, video_collision_ti_t8, keyframe_t8, id_t8);

%trial 9
[kfstats09, kfevents09] = readkeyframefile( 'keyframe_09' );
keyframe_t9 = [ 1.11, 1.23, 2.16, 2.27, 3.10, 3.24, 4.06, 4.22, 5.02, 5.14, 5.25, 6.08, 6.21, 8.22, 9.06, 9.17, 9.28, 10.09, 10.23 ];
id_t9 = [1,2,4,5,6,7,8,9,10,11,12,13,14,19,20,21,22,23,24 ];
% lost 3,15,16,17,18?
vicon_collision_ti_t9 = 0.778;
vicon_collision_fixedstep_ti_t9 = 0.78;         % this is assuming vicon is at fixed framerate
video_collision_ti_t9 = 1.24;
validateframedata(kfstats09, kfevents09, vicon_collision_ti_t9, vicon_collision_fixedstep_ti_t9, video_collision_ti_t9, keyframe_t9, id_t9);

%trial 10
[kfstats10, kfevents10] = readkeyframefile( 'keyframe_10' );
keyframe_t10 = [1.05, 1.18, 1.29, 2.11, 2.22, 5.05, 5.17, 6.00, 6.13, 6.24, 7.06, 7.18, 8.12, 9.10, 10.18, 11.01, 11.15, 11.26, 12.07 ];
id_t10 = [1,2,3,4,5,11,12,13,14,15,16,17,19,21,24,25,26,27,28 ];
% lost 6,7,8,9,10,18,20?,22,23
vicon_collision_ti_t10 = 1.958;
vicon_collision_fixedstep_ti_t10 = 1.96;         % this is assuming vicon is at fixed framerate
video_collision_ti_t10 = 3.2;
validateframedata(kfstats10, kfevents10, vicon_collision_ti_t10, vicon_collision_fixedstep_ti_t10, video_collision_ti_t10, keyframe_t10, id_t10);