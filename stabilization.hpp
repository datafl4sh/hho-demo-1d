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
class stabilization_operator
{
    size_t          m_degree;
    
    arma::Mat<T>    mass_matrix;
    arma::Mat<T>    stab_matrix;
    basis<T>        m_basis;
    quadrature<T>   m_quad;
    
    void
    build_matrices(const element<T>& elem, const arma::Mat<T>& gradrec_matrix)
    {
        mass_matrix.resize(m_basis.size(), m_basis.size());
        mass_matrix.zeros();
        
        auto qd = m_quad.integrate(elem);
        for (auto& qp : qd)
        {
            auto qpoint  = qp.first;
            auto qweight = qp.second;
            
            auto phi = m_basis.eval_functions(elem, qpoint);
            
            mass_matrix += qweight * phi * phi.t();
        }
        
        auto basis_k_size = m_basis.size() - 1;
        auto blocksz = arma::size(basis_k_size, basis_k_size);
        arma::Mat<T> M1 = mass_matrix.submat(0,0,blocksz);
        arma::Mat<T> M2 = mass_matrix.submat(0,1,blocksz);
        arma::Mat<T> proj1 = -solve(M1, M2*gradrec_matrix);
        arma::Mat<T> I_T(basis_k_size, basis_k_size);
        I_T.eye();
        proj1.submat(0,0,blocksz) += I_T;
        
        auto faces = elem.faces();
        
        auto phiF1 = m_basis.eval_functions(elem, faces[0]);
        auto dphiF1 = m_basis.eval_gradients(elem, faces[0]);
        auto phiF2 = m_basis.eval_functions(elem, faces[1]);
        auto dphiF2 = m_basis.eval_gradients(elem, faces[1]);
        
        arma::Mat<T> MFF, proj2, proj3, B;
        arma::Col<T> MFT;
        auto grads_size = m_basis.size() - 1;
        auto h = elem.measure();
        
        MFF = 1;//phiF1(0) * phiF1(0); //actually it's a scalar
        MFT = phiF1;
        proj2 = solve(MFF, MFT.tail(grads_size).t()*gradrec_matrix);
        proj2(0, basis_k_size) -= 1;
        proj3 = solve(MFF, MFT.head(basis_k_size).t() * proj1);
        B = proj2 + proj3;
        stab_matrix = B.t() * MFF * B / h;
        
        MFF = 1;//phiF2(0) * phiF2(0); //actually it's a scalar
        MFT = phiF2;
        proj2 = solve(MFF, MFT.tail(grads_size).t()*gradrec_matrix);
        proj2(0, basis_k_size+1) -= 1;
        proj3 = solve(MFF, MFT.head(basis_k_size).t() * proj1);
        B = proj2 + proj3;
        stab_matrix += B.t() * MFF * B / h;
    }
    
public:
    stabilization_operator()
        : m_degree(1)
    {
        m_basis = basis<T>(2);
        m_quad = quadrature<T>(4);
    }
    
    stabilization_operator(size_t degree)
        : m_degree(degree)
    {
        m_basis = basis<T>( m_degree+1 );
        m_quad = quadrature<T>( 2*(m_degree+1) );
    }
    
    void
    build(const element<T>& elem, const arma::Mat<T>& gradrec_matrix)
    {
        build_matrices(elem, gradrec_matrix);
    }
    
    arma::Mat<T>
    local_contrib(void) const
    {
        return stab_matrix;
    }
};


