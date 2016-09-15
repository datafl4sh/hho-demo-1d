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

__On Debian-derived Linux systems:__

    apt-get install libarmadillo-dev libboost-all-dev gnuplot
    
__On RedHat-derived Linux systems:__

	yum install armadillo-devel boost-devel gnuplot

__On Mac OS X__

    brew install armadillo boost gnuplot
    
Then, no matter on which system you are
	
	c++ -O3 -std=c++11 -o hho-demo-1d hho-1d-demo.cpp \
	        -larmadillo -lboost_system -lboost_iostreams
           
On Mac OS X you have also a second route:

    brew tap datafl4sh/code
    brew install hho-demo-1d
    
__On Windows__

It is possible to compile and run this code also on Windows by using [CygWin](https://cygwin.com). Using the CygWin package manager install the following packages:
 
    g++ make cmake git wget blas lapack libboost-devel gnuplot xorg-server xinit
    
Open the CygWin terminal and

    git clone https://github.com/datafl4sh/hho-demo-1d.git
    wget http://sourceforge.net/projects/arma/files/armadillo-7.400.2.tar.xz
    tar -Jxvf armadillo-7.400.2.tar.xz
    
Finally
    
    cd hho-demo-1d
    g++ -O3 -std=gnu++11 -I../armadillo-7.400.2/include \
            -o hho-demo-1d hho-1d-demo.cpp \
	        -lblas -llapack -lboost_system -lboost_iostreams
	        
To run, first you have to start (only one time) the CygWin X Windows System by

    startxwin&
    
Then you can launch the code with (see below)
    
    DISPLAY=:0 ./hho-demo-1d.exe [options] <example>
    
Many thanks to [Omar](http://blog.solidspace.org) for providing the Windows machine where I tested this procedure!

Running the code
----------------

The program supports a number of command line switches. The general syntax is

    ./hho-demo-1d [options] <example>
    
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
