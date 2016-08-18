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
#include "basis.hpp"
#include "quadrature.hpp"

template<typename T>
class projector
{
    basis<T>        m_basis;
    quadrature<T>   m_quad;
    size_t          m_degree;
    
public:
    projector()
        : m_degree(1)
    {}
    
    projector(size_t degree)
        : m_degree(degree)
    {}
    
    template<typename Function>
    arma::Col<T>
    project(const element<T>& elem, const Function& f)
    {
        m_basis = basis<T>(m_degree);
        m_quad  = quadrature<T>(2*m_degree);
     
        arma::Mat<T>    mass_matrix(m_basis.size(), m_basis.size());
        arma::Col<T>    rhs(m_basis.size());
        
        mass_matrix.zeros();
        rhs.zeros();
        
        auto qd = m_quad.integrate(elem);
        for (auto& qp : qd)
        {
            auto qpoint  = qp.first;
            auto qweight = qp.second;
            
            auto phi = m_basis.eval_functions(elem, qpoint);
            mass_matrix += qweight * phi * phi.t();
            rhs += qweight * phi * f(qpoint);
        }
        
        return solve(mass_matrix, rhs);
    }
    
    template<typename Function>
    arma::Col<T>
    rhs(const element<T>& elem, const Function& f)
    {
        m_basis = basis<T>(m_degree);
        m_quad  = quadrature<T>(2*m_degree);
        
        arma::Col<T>    r_rhs(m_basis.size());
        
        r_rhs.zeros();
        
        auto qd = m_quad.integrate(elem);
        for (auto& qp : qd)
        {
            auto qpoint  = qp.first;
            auto qweight = qp.second;
            
            auto phi = m_basis.eval_functions(elem, qpoint);
            r_rhs += qweight * phi * f(qpoint);
        }
        
        return r_rhs;
    }
    
    arma::Mat<T>
    as_matrix(const element<T>& elem)
    {
        m_basis = basis<T>(m_degree);
        m_quad  = quadrature<T>(2*m_degree);
        
        arma::Mat<T> mass_matrix(m_basis.size(), m_basis.size());
        mass_matrix.zeros();
        
        auto qd = m_quad.integrate(elem);
        for (auto& qp : qd)
        {
            auto qpoint  = qp.first;
            auto qweight = qp.second;
            
            auto phi = m_basis.eval_functions(elem, qpoint);
            mass_matrix += qweight * phi * phi.t();
        }
        
        return mass_matrix;
    }
    
    T
    eval_projection(const element<T>& elem, const arma::Col<T>& projection, T point)
    {
        auto phi = m_basis.eval_functions(elem, point);
        return dot(phi, projection);
    }
};

