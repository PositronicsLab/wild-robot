
% populate and instantiate trial 01
trial_id = 1;
keyframes = [1.29, 2.13, 2.24, 3.08, 3.22, 4.05, 4.17, 5.0, 5.26, 6.11, 8.26, 8.28, 9.26, 10.09, ...
            10.21, 11.06, 11.20, 12.05, 12.16, 13.0, 13.12, 13.28, 14.14, 14.27, 15.09, ...
            15.22, 16.04, 16.17, 17.02];
ids = [1,2,3,4,5,6,7,8,10,11,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35];
vicon_sync_ts = 1.33;
vicon_sync_t = 1.33;
video_sync_kf = 3.0;

trial01 = fusiondata( trial_id, keyframes, ids, vicon_sync_ts, vicon_sync_t, video_sync_kf );
%trial01 = trial01.fit_motor_frequency(true);
trial01 = trial01.fit_motor_frequency(false);
trial01 = trial01.interpolate_initial_internal_state;
trial01 = trial01.parse_raw_vicon_data;
% trial01.export_fused_sample_data;

% % populate and instantiate trial 02
% trial_id = 2;
% keyframes = [1.10, 1.25, 2.06, 2.19, 3.01, 3.17, 4.00, 5.25, 6.11, 6.24, 7.05, 7.18, 8.00, 8.13, 9.14, 10.10 ];
% ids = [1,2,3,4,5,6,7,11,12,13,14,15,16,17,20,22];
% % lost 8, 9, 10, 11, 18, 19?
% vicon_sync_ts = 1.869;
% vicon_sync_t = 1.87;         % this is assuming vicon is at fixed framerate
% video_sync_kf = 2.27;
% 
% trial02 = fusiondata( trial_id, keyframes, ids, vicon_sync_ts, vicon_sync_t, video_sync_kf );
% %trial02 = trial02.fit_motor_frequency(true);
% trial02 = trial02.fit_motor_frequency(false);
% trial02 = trial02.interpolate_initial_internal_state;
% trial02 = trial02.parse_raw_vicon_data;
% trial02.export_fused_sample_data;
% 
% 
% % populate and instantiate trial 03
% trial_id = 3;
% keyframes = [3.07, 3.23, 4.05, 4.17, 4.29, 5.13, 5.29, 6.24, 7.05, 7.20, 8.05, 8.21, 9.05, 9.18, 10.01, 10.28, 11.14, 12.09 ];
% ids = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,18,19,21];
% % lost 8, 17, 20
% vicon_sync_ts = 2.062;    
% vicon_sync_t = 1.88;         % this is assuming vicon is at fixed framerate
% video_sync_kf = 5.0;
% 
% trial03 = fusiondata( trial_id, keyframes, ids, vicon_sync_ts, vicon_sync_t, video_sync_kf );
% %trial03 = trial03.fit_motor_frequency(true);
% trial03 = trial03.fit_motor_frequency(false);
% trial03 = trial03.interpolate_initial_internal_state;
% trial03 = trial03.parse_raw_vicon_data;
% trial03.export_fused_sample_data;
% 
% 
% % populate and instantiate trial 04
% trial_id = 4;
% keyframes = [ 1.17, 2.02, 2.13, 2.27, 3.11, 5.18, 6.01, 6.15, 7.16, 7.28, 8.11, 8.23, 9.21, 10.03, 10.20, 11.03, 11.16, 12.02 ];
% ids = [1,2,3,4,5,10,11,13,14,15,16,18,19,20,22,23,24,25];
% % lost 6,7,8,9,10?,12,17,21
% vicon_sync_ts = 1.481;
% vicon_sync_t = 1.48;         % this is assuming vicon is at fixed framerate
% video_sync_kf = 3.1;
% 
% trial04 = fusiondata( trial_id, keyframes, ids, vicon_sync_ts, vicon_sync_t, video_sync_kf );
% %trial04 = trial04.fit_motor_frequency(true);
% trial04 = trial04.fit_motor_frequency(false);
% trial04 = trial04.interpolate_initial_internal_state;
% trial04 = trial04.parse_raw_vicon_data;
% trial04.export_fused_sample_data;
% 
% 
% % populate and instantiate trial 05
% trial_id = 5;
% keyframes = [1.02, 1.14, 1.27, 2.09, 2.21, 3.09, 4.18, 5.02, 5.15, 5.28, 6.11, 6.28, 7.10, 7.23, 8.05, 8.17, 9.17, 10.12, 10.24  ];
% ids = [1,2,3,4,5,6,10,11,12,13,14,15,16,17,18,19,21,23,24 ];
% % lost 7,8,9, 20, 22
% vicon_sync_ts = 1.656;
% vicon_sync_t = 1.66;         % this is assuming vicon is at fixed framerate
% video_sync_kf = 2.8;
% 
% trial05 = fusiondata( trial_id, keyframes, ids, vicon_sync_ts, vicon_sync_t, video_sync_kf );
% %trial05 = trial05.fit_motor_frequency(true);
% trial05 = trial05.fit_motor_frequency(false);
% trial05 = trial05.interpolate_initial_internal_state;
% trial05 = trial05.parse_raw_vicon_data;
% trial05.export_fused_sample_data;
% 
% 
% % populate and instantiate trial 06
% trial_id = 6;
% keyframes = [2.03, 2.17, 2.29, 3.11, 3.23, 4.07, 4.22, 5.19, 6.01, 6.14, 7.00, 7.13, 8.00, 8.15, 8.28, 9.11, 9.23, 10.22, 11.07 ];
% ids = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,20,21 ];
% % lost 8,19
% vicon_sync_ts = 2.219;
% vicon_sync_t = 2.22;         % this is assuming vicon is at fixed framerate
% video_sync_kf = 4.3;
% 
% trial06 = fusiondata( trial_id, keyframes, ids, vicon_sync_ts, vicon_sync_t, video_sync_kf );
% %trial06 = trial06.fit_motor_frequency(true);
% trial06 = trial06.fit_motor_frequency(false);
% trial06 = trial06.interpolate_initial_internal_state;
% trial06 = trial06.parse_raw_vicon_data;
% trial06.export_fused_sample_data;
% 
% 
% % populate and instantiate trial 01
% trial_id = 7;
% keyframes = [1.13, 1.25, 2.07, 2.21, 3.01, 3.13, 3.25, 4.06, 4.18, 4.29, 5.12, 7.21, 8.04, 8.14, 8.26, 9.05, 9.21, 10.03, 10.15];
% ids = [1,2,3,4,5,6,7,8,9,10,11,17,18,19,20,21,23,24,25];
% % lost 12,13,14,15,16,17?,22?
% vicon_sync_ts = 2.231;
% vicon_sync_t = 1.89;         % this is assuming vicon is at fixed framerate
% video_sync_kf = 3.5;
% 
% trial07 = fusiondata( trial_id, keyframes, ids, vicon_sync_ts, vicon_sync_t, video_sync_kf );
% %trial07 = trial07.fit_motor_frequency(true);
% trial07 = trial07.fit_motor_frequency(false);
% trial07 = trial07.interpolate_initial_internal_state;
% trial07 = trial07.parse_raw_vicon_data;
% trial07.export_fused_sample_data;
% 
% 
% % populate and instantiate trial 08
% trial_id = 8;
% keyframes = [1.08, 1.18, 2.00, 2.13, 2.28, 3.09, 3.21, 4.02, 4.27, 5.10, 5.21, 6.02, 6.14, 6.28, 7.10, 10.18, 10.28 ];
% ids = [1,2,3,4,5,6,7,8,10,11,12,13,14,15,16,24,25 ];
% % lost 9,17,18,19,20,21,22,23?
% vicon_sync_ts = 2.105;
% vicon_sync_t = 2.11;         % this is assuming vicon is at fixed framerate
% video_sync_kf = 2.4;
% 
% trial08 = fusiondata( trial_id, keyframes, ids, vicon_sync_ts, vicon_sync_t, video_sync_kf );
% %trial08 = trial08.fit_motor_frequency(true);
% trial08 = trial08.fit_motor_frequency(false);
% trial08 = trial08.interpolate_initial_internal_state;
% trial08 = trial08.parse_raw_vicon_data;
% trial08.export_fused_sample_data;
% 
% % populate and instantiate trial 09
% trial_id = 9;
% keyframes = [ 1.11, 1.23, 2.16, 2.27, 3.10, 3.24, 4.06, 4.22, 5.02, 5.14, 5.25, 6.08, 6.21, 8.22, 9.06, 9.17, 9.28, 10.09, 10.23 ];
% ids = [1,2,4,5,6,7,8,9,10,11,12,13,14,19,20,21,22,23,24 ];
% % lost 3,15,16,17,18?
% vicon_sync_ts = 0.778;
% vicon_sync_t = 0.78;         % this is assuming vicon is at fixed framerate
% video_sync_kf = 1.24;
% 
% trial09 = fusiondata( trial_id, keyframes, ids, vicon_sync_ts, vicon_sync_t, video_sync_kf );
% %trial09 = trial09.fit_motor_frequency(true);
% trial09 = trial09.fit_motor_frequency(false);
% trial09 = trial09.interpolate_initial_internal_state;
% trial09 = trial09.parse_raw_vicon_data;
% trial09.export_fused_sample_data;
% 
% 
% % populate and instantiate trial 10
% trial_id = 10;
% keyframes = [1.05, 1.18, 1.29, 2.11, 2.22, 5.05, 5.17, 6.00, 6.13, 6.24, 7.06, 7.18, 8.12, 9.10, 10.18, 11.01, 11.15, 11.26, 12.07 ];
% ids = [1,2,3,4,5,11,12,13,14,15,16,17,19,21,24,25,26,27,28 ];
% % lost 6,7,8,9,10,18,20?,22,23
% vicon_sync_ts = 1.958;
% vicon_sync_t = 1.96;         % this is assuming vicon is at fixed framerate
% video_sync_kf = 3.2;
% 
% trial10 = fusiondata( trial_id, keyframes, ids, vicon_sync_ts, vicon_sync_t, video_sync_kf );
% %trial10 = trial10.fit_motor_frequency(true);
% trial10 = trial10.fit_motor_frequency(false);
% trial10 = trial10.interpolate_initial_internal_state;
% trial10 = trial10.parse_raw_vicon_data;
% trial10.export_fused_sample_data;
