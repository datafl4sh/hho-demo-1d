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

#include <armadillo>

#include "element.hpp"
#include "quadrature.hpp"
#include "basis.hpp"
#include "stabilization.hpp"
#include "conjugate_gradient.hpp"

template<typename T>
using spmat_tuple = std::tuple<size_t, size_t, T>;


/****************************************************************************************
 * Example 3: diffusion
 *
 */

template<typename T>
std::vector<element<T>>
generate_mesh(size_t num_elements)
{
    std::vector<element<T>> mesh;
    mesh.reserve(num_elements);
    for (size_t i = 0; i < num_elements; i++)
    {
        element<T> elem(T(i)/num_elements, T(i+1)/num_elements);
        mesh.push_back(elem);
    }
    
    return mesh;
}

template<typename T, typename Function>
arma::Col<T>
solve_diffusion_problem(const run_parameters& rp, const Function& pf,
                        const std::vector<element<T>>& mesh)
{
    size_t basis_k_size     = rp.degree + 1;
    size_t dofs_num         = rp.num_elements + 3;
    
    std::vector<spmat_tuple<T>> tuples;
    arma::Col<T> sysrhs(dofs_num);
    sysrhs.zeros();
    
    projector<T>                            proj(rp.degree);
    gradient_reconstruction_operator<T>     gr(rp.degree);
    stabilization_operator<T>               stab(rp.degree);
    
    size_t elem_num = 0;
    for (auto& elem : mesh)
    {
        /* Compute projection on current element */
        auto projection = proj.rhs(elem, pf);
        
        gr.build(elem);
        stab.build(elem, gr.as_matrix());
        
        arma::Mat<T> A = gr.local_contrib();
        arma::Mat<T> S = stab.local_contrib();
        
        arma::Mat<T> LC = A + S;
        
        arma::Mat<T> K_TT = LC.submat(0, 0, arma::size(basis_k_size, basis_k_size));
        arma::Mat<T> K_TF = LC.submat(0, basis_k_size, arma::size(basis_k_size, 2));
        arma::Mat<T> K_FT = LC.submat(basis_k_size, 0, arma::size(2, basis_k_size));
        arma::Mat<T> K_FF = LC.submat(basis_k_size, basis_k_size, arma::size(2, 2));
        
        arma::Mat<T> AL = solve(K_TT, K_TF);
        arma::Col<T> bL = solve(K_TT, projection);
        
        arma::Mat<T> AC = K_FF - K_FT * AL;
        arma::Col<T> bC = /* explain this */ - K_FT * bL;
        
        for (size_t i = 0; i < AC.n_rows; i++)
        {
            for (size_t j = 0; j < AC.n_cols; j++)
                tuples.push_back( std::make_tuple(elem_num+i, elem_num+j, AC(i,j)) );
            sysrhs(elem_num+i) += bC(i);
        }
        
        elem_num++;
    }
    
    tuples.push_back( std::make_tuple(0, dofs_num-2, 1) );
    tuples.push_back( std::make_tuple(dofs_num-3, dofs_num-1, 1) );
    tuples.push_back( std::make_tuple(dofs_num-2, 0, 1) );
    tuples.push_back( std::make_tuple(dofs_num-1, dofs_num-3, 1) );
    
    arma::umat      locations(2, tuples.size());
    arma::Col<T>    values(tuples.size());
    
    for (size_t i = 0; i < tuples.size(); i++)
    {
        locations(0,i) = std::get<0>(tuples[i]);
        locations(1,i) = std::get<1>(tuples[i]);
        values(i) = std::get<2>(tuples[i]);
    }
    
    arma::SpMat<T>  sysmat(true, locations, values, dofs_num, dofs_num);
    
    // CG is definitely not the right solver because of the way the boundary
    // conditions are imposed. However it appears to work, so we keep it for
    // now.
    return conjugate_gradient(sysmat, sysrhs, 1e-9, 2*sysmat.n_cols);
}


template<typename T, typename Function, typename AnalyticSolution>
std::tuple<arma::Col<T>, arma::Col<T>, T, T>
postprocess(const run_parameters& rp, const arma::Col<T>& x,
            const Function& pf, const AnalyticSolution& sf,
            const std::vector<element<T>>& mesh)
{
    size_t basis_k_size     = rp.degree + 1;
    
    arma::Col<T> x_val(rp.num_elements * rp.eval_per_elem);
    arma::Col<T> pot_val(rp.num_elements * rp.eval_per_elem);
    
    projector<T>                            proj(rp.degree);
    gradient_reconstruction_operator<T>     gr(rp.degree);
    stabilization_operator<T>               stab(rp.degree);
    quadrature<T>                           quad(2*rp.degree);
    basis<T>                                basis(rp.degree);
    size_t pos = 0;
    size_t elem_num = 0;
    T l2_err = 0.;
    T l2_err_func = 0.;
    for (auto& elem : mesh)
    {
        arma::Col<T> solF(2);
        solF(0) = x(elem_num);
        solF(1) = x(elem_num+1);
        
        gr.build(elem);
        stab.build(elem, gr.as_matrix());
        
        arma::Mat<T> A = gr.local_contrib();
        arma::Mat<T> S = stab.local_contrib();
        
        arma::Mat<T> LC = A + S;
        
        arma::Mat<T> K_TT = LC.submat(0, 0, arma::size(basis_k_size, basis_k_size));
        arma::Mat<T> K_TF = LC.submat(0, basis_k_size, arma::size(basis_k_size, 2));
        
        arma::Col<T> rhs_c = proj.rhs(elem, pf);
        arma::Col<T> solT = solve(K_TT, rhs_c - K_TF*solF);
        
        arma::Col<T> sol(basis_k_size+2);
        sol.head(basis_k_size) = solT;
        sol.tail(2) = solF;
        
        //std::cout << (gr.as_matrix() * sol).t() << std::endl;
        
        /* Compute some test points inside the element */
        auto tps = make_test_points(elem, rp.eval_per_elem);
        
        /* Postprocess: recover the solution on the test points */
        for (size_t j = 0; j < rp.eval_per_elem; j++)
        {
            x_val(pos) = tps[j];
            pot_val(pos) = gr.reconstruct_potential(elem, sol, tps[j]);
            pos++;
        }
        
        arma::Col<T> asolT = proj.project(elem, sf);
        arma::Mat<T> mass = proj.as_matrix(elem);
        
        arma::Mat<T> me = mass.submat(0, 0, arma::size(basis_k_size, basis_k_size));
        arma::Col<T> ve_t = (asolT - solT);
        arma::Col<T> ve = ve_t.head(basis_k_size);
        
        l2_err += dot( ve, me * ve );
        
        
        auto qd = quad.integrate(elem);
        for (auto& qp : qd)
        {
            auto qpoint  = qp.first;
            auto qweight = qp.second;
            
            auto fval = sf(qpoint);
            
            T rval = 0.;
            auto phi = basis.eval_functions(elem, qpoint);
            for (size_t i = 0; i < phi.size(); i++)
                rval += phi[i] * solT(i);
            
            l2_err_func += (rval-fval) * (rval-fval) * qweight;
        }
        
        elem_num++;
    }
    
    return std::make_tuple(x_val, pot_val, sqrt(l2_err), sqrt(l2_err_func));
}

template<typename T>
int
run_example_diffusion(const run_parameters& rp)
{
    auto pf = [](T x) -> T {
        return 3.141592*3.141592*sin(3.141592*x);
    };
    
    auto sf = [](T x) -> T {
        return sin(3.141592*x);
    };
 
    /* Generate mesh */
    auto mesh = generate_mesh<T>(rp.num_elements);
    
    auto x = solve_diffusion_problem(rp, pf, mesh);
    auto pp = postprocess(rp, x, pf, sf, mesh);
    
    std::cout << "Err (with dofs) = " << std::get<2>(pp) << std::endl;
    std::cout << "Err (with func) = " << std::get<3>(pp) << std::endl;
    std::cout << "Difference      = " << std::get<2>(pp) - std::get<3>(pp) << std::endl;
    
    if (rp.draw)
    {
        Gnuplot gp;
        gp << "set grid" << std::endl;
        gp << "plot '-' with lines title 'potential'" << std::endl;
        gp.send1d(std::make_pair(std::get<0>(pp), std::get<1>(pp)));
    }
    
    return 0;
    
}



