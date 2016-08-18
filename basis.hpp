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
#include <utility>

#include "element.hpp"

template<typename T>
class basis
{
    size_t  m_degree;
    
public:
    basis()
        : m_degree(1)
    {}
    
    basis(size_t degree)
        : m_degree(degree)
    {}
    
    arma::Col<T>
    eval_functions(const element<T>& elem, const T& point)
    {
        auto bar = elem.center();
        auto h = elem.measure();
        auto ep = (point - bar)/h;
        
        arma::Col<T> ret(m_degree+1);
        for(size_t i = 0; i < m_degree+1; i++)
            ret(i) = std::pow(ep, i);
        
        return ret;
    }
    
    arma::Col<T>
    eval_gradients(const element<T>& elem, const T& point)
    {
        auto bar = elem.center();
        auto h = elem.measure();
        auto ep = (point - bar)/h;
        
        arma::Col<T> ret(m_degree+1);
        ret(0) = 0.;
        for(size_t i = 1; i < m_degree+1; i++)
            ret(i) = (i/h)*std::pow(ep, i-1);
        
        return ret;
    }
    
    size_t size() const
    {
        return m_degree+1;
    }
    
    size_t degree_index(size_t degree) const
    {
        return degree;
    }
};

