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
#include <utility>
#include <algorithm>

#include "element.hpp"

template<typename T>
class quadrature
{
    std::vector<std::pair<T,T>>     m_quadrature_data;
    size_t                          m_order;
    
    void
    compute(size_t order)
    {
        size_t num_points;
        
        if (order%2 == 0)
            order++;
        
        num_points = std::max( ((order%2 == 0) ? order+1 : order+2)/2, size_t(1));
        
        if (num_points == 1)
        {
            m_quadrature_data.push_back( std::make_pair( 0., 1.) );
            return;
        }
        
        arma::Col<T> vals(num_points-1);
        for (size_t i = 0; i < vals.size(); i++)
            vals(i) = std::sqrt( 1. / ( 4. - 1. / ((i+1)*(i+1)) ) );
        
        arma::Mat<T> M(num_points, num_points);
        M.zeros();
        M.diag(-1) = vals;
        M.diag(+1) = vals;
        
        arma::Mat<T> eigvec;
        arma::Col<T> eigval;
        eig_sym(eigval, eigvec, M);
        
        arma::Row<T> weights = eigvec.row(0);
        
        m_quadrature_data.clear();
        m_quadrature_data.reserve(num_points);
        
        for (size_t i = 0; i < num_points; i++)
        {
            auto qd = std::make_pair(eigval(i), weights(i)*weights(i));
            m_quadrature_data.push_back( qd );
        }
    }
    
public:
    quadrature()
        : m_order(1)
    {
        compute(m_order);
    }
    
    quadrature(size_t order)
        : m_order(order)
    {
        compute(m_order);
    }
    
    std::vector<std::pair<T,T>>
    integrate(const element<T>& elem)
    {
        std::vector<std::pair<T,T>> ret;
        ret.resize( m_quadrature_data.size() );
        
        auto h = elem.measure();
        auto pts = elem.points();
        
        auto tf = [&](const std::pair<T,T>& qd) -> std::pair<T,T> {
            auto pt = (qd.first+1)*0.5*h + pts[0];
            auto w = qd.second * h;
            return std::make_pair(pt, w);
        };
        
        std::transform(m_quadrature_data.begin(), m_quadrature_data.end(),
                       ret.begin(), tf);
        
        return ret;
    }
};




