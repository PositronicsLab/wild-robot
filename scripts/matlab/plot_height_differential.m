h_raw = load( 'height_unfiltered.log' );
h_raw_lbl = 'raw vicon center';

h_min = load( 'height_corrected_by_min.log' );
h_min_lbl = 'minimized center';

h_vsk = load( 'height_corrected_by_vsk.log' );
h_vsk_lbl = 'perturbed center';

plot( h_raw(:,1), h_raw(:,4)-0.041 );
hold on
plot( h_vsk(:,1), h_vsk(:,4)-0.041, 'g' );
plot( h_min(:,1), h_min(:,4)-0.041, 'r' );
xlabel('virtual time (s)');
ylabel('vertical distance of vicon wb center from ideal center (m)');
legend(h_raw_lbl, h_vsk_lbl, h_min_lbl);
hold off