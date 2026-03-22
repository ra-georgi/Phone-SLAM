set terminal pngcairo size 1400,800
set output 'gyro_plot.png'
set grid
set key left top
set xlabel 'seconds_elapsed (s)'
set ylabel 'angular velocity (units in file)'
set title 'Gyroscope: x/y/z vs time'
plot 'gyro_plot.dat' using 1:2 with lines title 'wx (x)', 'gyro_plot.dat' using 1:3 with lines title 'wy (y)', 'gyro_plot.dat' using 1:4 with lines title 'wz (z)'
