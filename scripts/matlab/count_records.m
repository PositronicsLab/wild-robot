function [ count ] = count_records( fd )
%COUNT_RECORDS Summary of this function goes here
%   Detailed explanation goes here

    % assumes that the cursor points to the beginning of the file

    count = 0;
    
    % a record is 4 lines
    txt1 = fgetl( fd );
    while( ischar(txt1) ) 
      if ~ischar( txt1 )
        break;
      end
      
      txt2 = fgetl( fd );
      txt3 = fgetl( fd );
      txt4 = fgetl( fd );
      count = count + 1;
      txt1 = fgetl( fd );   
    end
    
    % return to the beginning of file
    fseek( fd, 0, 'bof' );
end

