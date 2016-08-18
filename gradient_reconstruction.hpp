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

template<typename T>
class gradient_reconstruction_operator
{
    size_t          m_degree;
    
    arma::Mat<T>    stiffness_matrix;
    arma::Mat<T>    gradrec_matrix;
    arma::Mat<T>    local_contrib_matrix;
    basis<T>        m_basis;
    quadrature<T>   m_quad;

    void
    build_matrices(const element<T>& elem)
    {
        stiffness_matrix.resize(m_basis.size(), m_basis.size());
        stiffness_matrix.zeros();
        
        auto qd = m_quad.integrate(elem);
        for (auto& qp : qd)
        {
            auto qpoint  = qp.first;
            auto qweight = qp.second;
            
            auto dphi = m_basis.eval_gradients(elem, qpoint);
            
            stiffness_matrix += qweight * dphi * dphi.t();
        }
        
        auto basis_k_size = m_basis.size() - 1;
        arma::Mat<T> MG = stiffness_matrix.submat(1,1, arma::size(basis_k_size, basis_k_size));
        
        auto bg_rows = m_basis.size() - 1;
        arma::Mat<T> BG;
        BG.resize(bg_rows, basis_k_size+2);
        
        auto blocksz = arma::size(bg_rows, basis_k_size);
        BG.submat(0,0,blocksz) = stiffness_matrix.submat(1,0,blocksz);
        
        auto faces = elem.faces();
        
        auto phiF1 = m_basis.eval_functions(elem, faces[0]);
        auto dphiF1 = m_basis.eval_gradients(elem, faces[0]);
        auto phiF2 = m_basis.eval_functions(elem, faces[1]);
        auto dphiF2 = m_basis.eval_gradients(elem, faces[1]);
        
        /* Beware of the signs: they are due to the normals */
        BG.submat(0,0,blocksz) += + dphiF1.tail(bg_rows) * phiF1.head(basis_k_size).t();
        BG.submat(0,0,blocksz) += - dphiF2.tail(bg_rows) * phiF2.head(basis_k_size).t();
        BG.col(basis_k_size)    = - dphiF1.tail(bg_rows); // * phiF1(0), but not needed, it is always 1
        BG.col(basis_k_size+1)  = + dphiF2.tail(bg_rows); // * phiF2(0), but not needed, it is always 1
        
        gradrec_matrix = solve(MG, BG);
        
        local_contrib_matrix = BG.t() * gradrec_matrix;
    }
    
public:
    gradient_reconstruction_operator()
        : m_degree(1)
    {
        m_basis = basis<T>(2);
        m_quad = quadrature<T>(4);
    }
    
    gradient_reconstruction_operator(size_t degree)
        : m_degree(degree)
    {
        m_basis = basis<T>( m_degree+1 );
        m_quad = quadrature<T>( 2*(m_degree+1) );
    }
    
    void
    build(const element<T>& elem)
    {
        build_matrices(elem);
    }
    
    T
    reconstruct_potential_zeroavg(const element<T>& elem, const arma::Col<T>& dofs, T point)
    {
        auto basis_size = m_basis.size() - 1;
        auto phi = m_basis.eval_functions(elem, point).tail(basis_size);
        return dot(phi, gradrec_matrix * dofs);
    }
    
    T
    reconstruct_potential(const element<T>& elem, const arma::Col<T>& dofs, T point)
    {
        auto basis_size = m_basis.size() - 1;
        auto phi = m_basis.eval_functions(elem, point).tail(basis_size);
        return dot(phi, gradrec_matrix * dofs) + dofs(0);
    }
    
    T
    reconstruct_gradient(const element<T>& elem, const arma::Col<T>& dofs, T point)
    {
        auto basis_size = m_basis.size() - 1;
        auto dphi = m_basis.eval_gradients(elem, point).tail(basis_size);
        return dot(dphi, gradrec_matrix * dofs);
    }
    
    arma::Mat<T>
    as_matrix(void) const
    {
        return gradrec_matrix;
    }
    
    arma::Mat<T>
    local_contrib(void) const
    {
        return local_contrib_matrix;
    }
    
};


