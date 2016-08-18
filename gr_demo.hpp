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

#pragma once

#include <vector>
#include <armadillo>

#include "gnuplot-iostream.h"

#include "common.h"

#include "element.hpp"
#include "quadrature.hpp"
#include "basis.hpp"
#include "projector.hpp"
#include "gradient_reconstruction.hpp"

/****************************************************************************************
 * Example 2: show what is the gradient reconstruction operator and what it does
 *
 */

template<typename T>
static int
run_example_gr(const run_parameters& rp)
{
    /* Generate mesh */
    std::vector<element<T>> mesh;
    mesh.reserve(rp.num_elements);
    for (size_t i = 0; i < rp.num_elements; i++)
    {
        element<T> elem(T(i)/rp.num_elements, T(i+1)/rp.num_elements);
        mesh.push_back(elem);
    }
    
    arma::Col<T> x_val(rp.num_elements * rp.eval_per_elem);
    arma::Col<T> grad_val(rp.num_elements * rp.eval_per_elem);
    arma::Col<T> pot_zeroavg_val(rp.num_elements * rp.eval_per_elem);
    arma::Col<T> pot_val(rp.num_elements * rp.eval_per_elem);
    
    size_t pos = 0;
    
    for (auto& elem : mesh)
    {
        /* Compute projection on current element */
        gradient_reconstruction_operator<T> gr(rp.degree);
        gr.build(elem);
        
        auto pf = [](T p) -> T {
            return sin(3.141592*p);
        };
        
        arma::Col<T> projection(rp.degree+3);
        projector<T> proj(rp.degree);
        projection.zeros();
        projection.head(rp.degree+1) = proj.project(elem, pf);
        
        auto faces = elem.faces();
        
        projection(rp.degree+1) = pf(faces[0]);
        projection(rp.degree+2) = pf(faces[1]);
        
        /* Compute some test points inside the element */
        auto tps = make_test_points(elem, rp.eval_per_elem);
        
        /* Postprocess: recover the solution on the test points */
        for (size_t j = 0; j < rp.eval_per_elem; j++)
        {
            x_val(pos) = tps[j];
            grad_val(pos) = gr.reconstruct_gradient(elem, projection, tps[j]);
            pot_zeroavg_val(pos) = gr.reconstruct_potential_zeroavg(elem, projection, tps[j]);
            pot_val(pos) = gr.reconstruct_potential(elem, projection, tps[j]);
            pos++;
        }
    }
    
    /* Plot it */
    if (rp.draw)
    {
        Gnuplot gp;
        gp << "set grid" << std::endl;
        gp << "set multiplot layout 3,1" << std::endl;
        gp << "plot '-' with lines title 'gradient'" << std::endl;
        gp.send1d(std::make_pair(x_val, grad_val));
        gp << "set grid" << std::endl;
        gp << "plot '-' with lines title 'potential (zeroavg)'" << std::endl;
        gp.send1d(std::make_pair(x_val, pot_zeroavg_val));
        gp << "set grid" << std::endl;
        gp << "plot '-' with lines title 'potential'" << std::endl;
        gp.send1d(std::make_pair(x_val, pot_val));
    }
    
    return 0;
}
