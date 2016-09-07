classdef fusiondata
    %FUSIONDATA Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
      % constants from the experimental setup
      video_fps        % HD video framerate
      vicon_hz         % vicon capture framerate
        
      % data gathered from each trial
      trial_id         % the trial identifier
      keyframes        % the keyframes when a signal was detected <seconds>.<frame_number>
      kf_ids           % the index of the signal corresponding to the keyframe vector
      vicon_sync_ts_t  % time of the first impact [from the system timestamp]
      vicon_sync_t     % time of the first impact assuming constant vicon framerate
      video_sync_kf    % keyframe, <second>.<frame_number>, of the first impact in video
      
      %NOTE the below 9-tuple is for pre processed data.  Really we want to
      %change this to the raw output data, so will need to change the
      %importer to support parsing
      vicon_samples    % the data acquired from the vicon system itself, a 10-tuple <vicon_t>,<vicon_ts>,<position[3]>,<rotation[4]>,<theta>
      % where vicon_t is computed based on the published frame_rate of data
      % capture and vicon_ts is the recorded system timestamp when a sample
      % was received in callback (the timestamp is adjusted to be the differential between the current timestamp and the initial timestamp)
      
      % derived from the data
      trial_name
      video_sync_t     % time of the first impact in video computed from keyframe and frame rate
      signals_t        % the time at which a signal was detected on video.  derived from keyframes
      motor_hz         % frequency of the motor derived from regression
      theta_t0         % initial orientation.  derived by interpolating back from initial signal
    end
    
    methods
      function obj = fusiondata( trial_id, keyframes, kf_ids, vicon_sync_ts_t, vicon_sync_t, video_sync_kf )
        obj.video_fps = 30;
        obj.vicon_hz = 100;
        obj.trial_id = trial_id;
        obj.keyframes = keyframes;
        obj.kf_ids = kf_ids;
        obj.vicon_sync_ts_t = vicon_sync_ts_t;
        obj.vicon_sync_t = vicon_sync_t;
        obj.video_sync_kf = video_sync_kf;
        obj.video_sync_t = keyframe_to_time( obj, video_sync_kf, obj.video_fps );
        obj.trial_name = strcat('trial_', int2str(trial_id));
      end
        
      function t = keyframe_to_time( obj, keyframe, fps )
        % converts a keyframe specified as <seconds>.<frame_number> to a
        % time in seconds.  Assumes that the frame_number is two digits and
        % in the interval [0,99]
        
        sec = floor(keyframe);
        frame = (keyframe - sec) * 100;
        t = sec + frame / fps;        
      end
      
      function s = timestamp_to_seconds( obj, min, sec, ms ) 
        s = double(min) * 60.0 + double(sec) + double(ms) / 1000.0;
      end
      
      function obj = fit_motor_frequency( obj, doplot )
        obj.signals_t = keyframe_to_time( obj, obj.keyframes, obj.video_fps );
        
        y = obj.signals_t;
        x = obj.kf_ids;

        n = size(x,2);

        % least squares regression
        p = polyfit( x, y, 1 );

        hz = 1/p(1);  % frequency of the signal computed from regression
        %bpm = hz*60;  % beats per minute for the signal 

        obj.motor_hz = hz;
        
        if doplot
          yfit = polyval(p,x);

          figure('Name', obj.trial_id );
          plot(x,y,'b')
          hold on
          plot(x,yfit,'r')
          hold off
        end
        %sim_t = y - delta;

        %fprintf( '%s: hz[%f], bpm[%f]\n', trial_name, hz, bpm );
      end
      
      function obj = interpolate_initial_internal_state( obj )
        % compute motor cycle time and motor velocity
        sec_per_cycle = 1/obj.motor_hz;
        omega = 2 * pi * obj.motor_hz;
        
        % time of initial signal
        t = obj.signals_t(1);
        t1 = t;
        while t > 0 
          t = t - sec_per_cycle;
          if t > 0
            t1 = t;
          end
        end

        % compute the initial orientation preceding the initial signal
        x1 = 2 * pi;             % 'zero' orientation at signal
        t0 = 0;                  % time to go back to
        dt = t1 - t0;            % change in time between initial orientation and 'zero' orientation

        % compute the change in orientation
        dtheta = omega * dt;

        % back out the change from 'zero' to find the initial orientation
        obj.theta_t0 = x1 - dtheta;
      end
      
      function filename = get_filename( obj )
        %filename = strcat( strcat('trial_', sprintf('%02d',obj.trial_id)),'.log');
        filename = strcat( strcat('trial', int2str(obj.trial_id)),'.txt');
      end
      
      function obj = load_vicon_data( obj ) 
        filename = strcat( strcat('trial_', sprintf('%02d',obj.trial_id)),'.log');
        
        obj.vicon_samples = load( filename );
      end
      
      function obj = parse_raw_vicon_data( obj )
        vicon_dt = 1 / obj.vicon_hz;
        vicon_t = 0.0;
        
        %filename = strcat( strcat('trial_', sprintf('%02d',obj.trial_id)),'.log')
        filename = get_filename( obj )
        fid = fopen( filename )
        first_record = true;
        vicon_ts0 = 0.0;
        while ~feof(fid)
          line1 = fgetl( fid );  % timestamp <%s: min:sec:ms>  Note %s contains several :'s
          line2 = fgetl( fid );  % position  <%s: x, y, z>
          line3 = fgetl( fid );  % quaternion <%s: x, y, z, w>
          line4 = fgetl( fid );  % empty line
          
          A = textscan( line1, '%s %s %s %d %d %d', 'Delimiter', ':' );
          B = textscan( line2, '%s %f %f %f', 'Delimiter', ':,' );
          C = textscan( line3, '%s %f %f %f %f', 'Delimiter', ':,' );

          ts = obj.timestamp_to_seconds( int32(A{4}), int32(A{5}), int32(A{6}) );
          pos = [double(B{2}), double(B{3}), double(B{4})];
          rot = [double(C{2}), double(C{3}), double(C{4}), double(C{5})]; 
          
          if first_record
            vicon_ts0 = ts;
            first_record = false;
          else
            vicon_t = vicon_t + vicon_dt;
          end
          ts = ts - vicon_ts0;
       
          % compute joint angle theta
          omega = 2 * pi * obj.motor_hz;     % assuming constant motor velocity
          theta = vicon_t * omega + obj.theta_t0;
          theta = theta - (2 * pi * floor( theta / (2 * pi) ) );
          
          obj.vicon_samples = [obj.vicon_samples;vicon_t,ts,pos,rot,theta];
        end
        fclose(fid);
      end
      function export_fused_sample_data( obj ) 
        filename = strcat( 'wb_fused_trial_', sprintf('%02d.txt', obj.trial_id) );
        fid = fopen( filename, 'w' );
        [rows, cols] = size(obj.vicon_samples);
        format = '%f %f %f %f %f %f %f %f %f %f\n';
        for row = 1:rows
          fprintf( fid, format, obj.vicon_samples(row,:) );
        end
        fclose( fid );
      end
    end
end

