function [ stats, events ] = readkeyframefile( filename )
%READKEYFRAME Summary of this function goes here
%   Detailed explanation goes here

    fid = fopen( filename );
    assert( ~feof(fid) );
    header1 = fgetl( fid );
    assert( ~feof(fid) );
    line = fgetl( fid ); 
    stats = textscan( line, '%f %f %f %f', 'Delimiter', ' ' );
    assert( ~feof(fid) );
    header2 = fgetl( fid );

    events = [];
    while ~feof( fid )
    	line = fgetl( fid );
        e = textscan( line, '%d %d %d', 'Delimiter', ' ' );
        events = [events; e];
    end
    stats = cell2mat(stats);
    events = cell2mat(events);
end

