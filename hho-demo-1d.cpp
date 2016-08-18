/*
 *       /\
 *      /__\        Matteo Cicuttin (C) 2016 - matteo.cicuttin@enpc.fr
 *     /_\/_\
 *    /\    /\      École Nationale des Ponts et Chaussées
 *   /__\  /__\     CERMICS
 *  /_\/_\/_\/_\
 *
 * This is a simple 1D code for the demonstration of the Hybrid High Order
 * numerical method. It is intended only to show the details of all the
 * involved operators (projection, gradient reconstruction, stabilization)
 * step by step.
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <iostream>
#include <cstdlib>
#include <unistd.h>

#include "common.h"



#include "projector_demo.hpp"
#include "gr_demo.hpp"
#include "diffusion_demo.hpp"

static void
usage(char *progname)
{
    std::cout << " -k <degree>      Polynomial order. Default = 1." << std::endl;
    std::cout << " -n <gridelem>    Number of grid elements. Default = 2." << std::endl;
    std::cout << " -p <numpts>      Number of evaluation points per element. Default = 5." << std::endl;
    std::cout << " -f <filename>    Name of the solution output file." << std::endl;
    std::cout << " -h               Print this help." << std::endl;
    
}

int
main(int argc, char **argv)
{
    using RealType = double;
    
    struct run_parameters rp;
    rp.filename         = nullptr;
    rp.draw             = false;
    rp.degree           = 1;
    rp.num_elements     = 2;
    rp.eval_per_elem    = 5;
    
    int ch;
    
    while ( (ch = getopt(argc, argv, "dhk:n:f:p:")) != -1 )
    {
        switch(ch)
        {
            case 'd':
                rp.draw = true;
                break;
                
            case 'k':
                rp.degree = atoi(optarg);
                if (rp.degree < 0)
                {
                    std::cout << "Degree must be positive. Falling back to 1." << std::endl;
                    rp.degree = 1;
                }
                break;
                
            case 'n':
                rp.num_elements = atoi(optarg);
                break;
                
            case 'p':
                rp.eval_per_elem = atoi(optarg);
                break;
                
            case 'f':
                rp.filename = optarg;
                break;
                
            case 'h':
            case '?':
            default:
                usage(argv[0]);
                exit(1);
        }
    }
    
    argc -= optind;
    argv += optind;
    
    
    if (argc == 0)
    {
        std::cout << "Please select an example" << std::endl;
        exit(1);
    }
    
    std::cout << "Running with the following parameters:" << std::endl;
    std::cout << "  K = " << rp.degree << std::endl;
    std::cout << "  N = " << rp.num_elements << std::endl;
    std::cout << "  output filename = " << (rp.filename == nullptr ? "(none)" : rp.filename) << std::endl;
    
    if ( strcmp(argv[0], "projection") == 0 )
        return run_example_projection<RealType>(rp);
    
    if ( strcmp(argv[0], "gradrec") == 0 )
        return run_example_gr<RealType>(rp);
    
    if ( strcmp(argv[0], "diffusion") == 0 )
        return run_example_diffusion<RealType>(rp);
    
    
    return 0;
}






