doplots = true;
range = 10:10;

x = linspace( 0, 4*pi, 100 );
y1 = sin( x );
%plot( x, y1 )
%figure

keys = []
for i = range
    filename = strcat( 'keyframe_', sprintf('%02d', i) );
    data = video_data( );
    data = data.read_data( filename );
    
    % get the time of all the samples
    time = data.samples(:,1);
    % get the index of all the samples
    index = data.samples(:,2);
    % compute the number of samples
    n = size( index,1 );
    % least squares regression to the time and index
    p = polyfit( index, time, 1 );
    
    if doplots
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
    end
        
    hz = 1/p(1);  % frequency of the signal computed from regression
    bpm = hz*60;  % beats per minute for the signal - sanity check by metronome - approximate bpm of 132-133 bpm
    
    fprintf( '%s : hz[%f], bpm[%f]\n', filename, hz, bpm );
 
    %key = strcat( 'trial ', sprintf('%02d', i) );
    %keys = [keys; key];
    %hold on
    %y2 = sin( x * hz );
    %plot( x, y2, 'Color', [rand,rand,rand] )
    %hold off
end

%legend( keys )