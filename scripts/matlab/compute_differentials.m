
for id = 1:10
  session = id;
  load( sprintf( 'V%02d.mat', session ) );
  set = eval(genvarname( sprintf('V%02d', session) ) );

  Q = zeros( size(set, 1) - 1, 13 );
  Q(:,1:9) = set(1:size(set,1)-1,1:9);

  n = size(set,1);

  for i = 1:n-1
    Q(i,10:12) = set( i+1, 3:5 ) - set( i, 3:5 );
    Q(i,13) = quatangle( set( i+1, 6:9 ), set( i, 6:9 ) );
  end

  state_var = genvarname( sprintf('q%02d', session) );
  assignin( 'base', state_var, Q );
end