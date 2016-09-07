function x = load_raw_vicon( r0, r1 )
%LOAD_RAW_VICON Summary of this function goes here
%   Detailed explanation goes here

  range = r0:r1;
  counts = zeros(r1 - r0 + 1,1);
  files = [];
  
  i = 1;
  for id = range
    files = [files; sprintf( 'trial_%02d.txt', id )];
    fd = fopen( files(i,:) );
    counts(i,1) = count_records( fd );
    fclose( fd );
    i = i + 1;
  end
  
  %disp(sum(counts));
  
  % allocate a buffer to hold the whole data set
  % record structure: session, frame-id, pos[x,y,z], rot[x,y,z,w]
  x = zeros(sum(counts), 9);

  % iterate over all records and load into the buffer
  n = 0;
  id = r0;
  for i = 1:size(counts,1)
    file = files(i,:);
    count = counts(i,1);
    fd = fopen( file );
    for j = 1:count
      n = n + 1;
      x(n,:) = [id, j-1, get_next_record( fd )];
    end
    fclose( fd );
    id = id + 1;
  end
  
end

