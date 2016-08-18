Hybrid High Order 1D demo code
==============================

This is a simple 1D code implementing the Hybrid High Order numerical method. It is intended to teach the fundamentals of the method. In particular, all the components are implemented in a self-contained way, so they can be studied separately. The components include:
 
 * Basis functions
 * Quadrature rules
 * Projector operator
 * Gradient reconstruction operator
 * Stabilization operator

All the components are then used to solve a simple 1D diffusion problem (Poisson equation).

Building the code
-----------------

To build the code you must have installed Armadillo, Boost and Gnuplot.

On Debian-derived Linux systems:

    apt-get install libarmadillo-dev libboost-all-dev gnuplot
    
On RedHat-derived Linux systems:

	yum install armadillo-devel boost-devel gnuplot

On Mac OS X you must have Homebrew installed, then

    brew install armadillo boost gnuplot
    
Then
	
	c++ -O3 -std=c++11 -o hho hho-1d-demo.cpp -larmadillo \
           -lboost_system -lboost_iostreams
      
Running the code
----------------

The program supports a number of command line switches. The general syntax is

    ./hho [options] <example>
    
The options are:

    -k <degree>      Polynomial order. Default = 1.
    -n <gridelem>    Number of grid elements. Default = 2.
    -p <numpts>      Number of evaluation points per element. Default = 5.
    -f <filename>    Name of the solution output file (not yet implemented).
    -h               Print the help.
    
The supported examples are

 * `projection`: demonstrates the usage of the projection operator
 * `gradrec`: demonstrates the gradient reconstruction operator
 * `diffusion`: solves an 1-dimensional diffusion problem
      
Have fun!
