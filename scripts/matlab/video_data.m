classdef video_data
    %VIDEO_DATA Summary of this class goes here
    %   Detailed explanation goes here
    
    properties
        framerate
        fps
        samples
        vicon_ti
        video_ti
    end
    
    methods
        function obj = video_data(  )
            obj.fps = 30;
            obj.framerate = 1/obj.fps;
            obj.samples = [];
        end
        function obj = read_data( obj, filename )
            [kfstats, kfevents] = readkeyframefile( filename );
            obj.fps = kfstats(1,4);
            obj.framerate = 1/obj.fps;

            %normalize video time (s + framenumber) to compatible double time
            [rows, cols] = size( kfevents );
            obj.samples = [];
            for row = 1:rows
                sec = cast( kfevents( row, 1 ), 'double' );
                frame = cast( kfevents( row, 2 ), 'double' );
                idx = cast( kfevents( row, 3 ), 'double' );
  
                video_t = sec + frame * obj.framerate;
                obj.samples = [obj.samples; [video_t, idx]];
            end
            %video_events now contains the map between approximate time (still in
            %camera temporal frame of referece) and the signal events

            %compute the error between vicon timestamps and video timestamps
            obj.vicon_ti = kfstats(1, 1);
            obj.video_ti = kfstats(1, 3);
            % the approximate differential between the video timestamp and the vicon
            % timestamp
            %diff_ti = video_ti - vicon_ti;
        end
        function obj = fit_frequency( obj )
            % get the time of all the samples
            time = obj.samples(1,:);
            % get the index of all the samples
            index = obj.samples(2,:);
            % compute the number of samples
            n = size( index, 1 );
            % least squares regression to the time and index
            p = polyfit( index, time, 1 );
    
            %if doplots
            % FOR PLOT ONLY - get the set of fitted time samples
                timefit = polyval(p,index);
    
                figure
                plot(time,index,'b')
                hold on
                plot(timefit,index,'r')
                hold off
                xlabel('time (s)')
                ylabel('event index')
                legend('sensor', 'lsqfit', 'Location', 'southeast')
            %end
        
            hz = 1/p(1);  % frequency of the signal computed from regression
            bpm = hz*60;  % beats per minute for the signal - sanity check by metronome - approximate bpm of 132-133 bpm
            fprintf( 'hz[%f], bpm[%f]\n', hz, bpm );
        end    
    end
end

