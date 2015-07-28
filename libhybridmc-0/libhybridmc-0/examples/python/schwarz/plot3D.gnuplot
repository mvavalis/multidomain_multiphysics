# set term qt
# set term qt 1

 set term png size 1000,700 enhanced font "Helvetica,20"
set term png enhanced
set output "xexact3D.png"

set terminal png font 'Verdana,24'
set key font ",24"
set xtics font "Verdana,24"

set logscale y
set style data lines

set title "Computed Solution Error"
set xlabel "Number of iterations (k)"
set ylabel "|| U_{true} - U_{k} ||"

plot "domain__sphere3D_1__iface__box3D_1.log" using 1:2 title "Sphere", \
"domain__box3D_1__iface__sphere3D_1.log" using 1:2 title "Box"

set output "xprev3D.png"
set title "Solution Convergence"
set xlabel "Number of iterations (k)"
set ylabel "|| U_{k} - U_{k-1} ||"

plot "domain__sphere3D_1__iface__box3D_1.log" every::1 using 1:3 title "Sphere", \
"domain__box3D_1__iface__sphere3D_1.log" using 1:3 title "Box"
