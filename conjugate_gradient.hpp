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

#include <iostream>
#include <armadillo>
#include <algorithm>
#include <cassert>

/* Trivial implementation of the unpreconditioned CG.
 * See "An Introduction to the Conjugate Gradient Method Without
 *      the Agonizing Pain" by J. R. Shewchuk
 */
template<typename T>
arma::Col<T>
conjugate_gradient(const arma::SpMat<T>& A, const arma::Col<T>& b, T eps = 1e-8, size_t maxit = 0)
{
    assert(A.n_cols == A.n_rows);
    
    arma::Col<T> d, r, x;
    T alpha, beta;
    T res, res0;
    
    maxit = std::max(maxit, size_t(A.n_cols));
    
    std::cout << "Starting CG. Target rr = " << eps << ", maxit = " << maxit << std::endl;
    
    x.resize(b.size());
    x.zeros();
    
    r = b - A*x;
    d = r;
    
    res = res0 = norm(r);
    
    size_t iter = 0;
    while ( (res/res0 > eps) and (iter++ < maxit) )
    {
        auto Ad = A * d;
        auto dot_rr = dot(r,r);
        
        alpha = dot_rr/dot(d, Ad);
        x = x + alpha * d;
        r = r - alpha * Ad;
        beta = dot(r,r)/dot_rr;
        d = r + beta * d;
        
        res = norm(r);
    }
    
    if ( (iter <= maxit) and (res/res0 < eps) )
    {
        std::cout << "Solver converged after " << iter;
        std::cout << " iterations, ||r||/||r0|| = " << res/res0;
        std::cout << std::endl;
    }
    else
    {
        std::cout << "Solver NOT converged! ||r||/||r0|| = " << res/res0 << std::endl;
    }
    
    return x;
}

