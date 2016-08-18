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

#include <array>

template<typename T>
class element
{
    T       m_p0, m_p1;
    
public:
    element()
        : m_p0(0), m_p1(0)
    {}
    
    element(T p0, T p1)
        : m_p0(p0), m_p1(p1)
    {}
    
    T center() const
    {
        return (m_p1+m_p0)/T(2);
    }
    
    T measure() const
    {
        return m_p1-m_p0;
    }
    
    std::array<T, 2>
    faces() const
    {
        return std::array<T, 2>({m_p0, m_p1});
    }
    
    std::array<T, 2>
    points() const
    {
        return std::array<T, 2>({m_p0, m_p1});
    }
    
    std::array<T, 2>
    normals() const
    {
        return std::array<T, 2>({T(-1), T(1)});
    }
};

template<typename T>
std::vector<T>
make_test_points(const element<T>& elem, size_t howmany)
{
    auto pts = elem.points();
    auto step = elem.measure()/(howmany-1);
    
    std::vector<T> ret;
    ret.reserve(howmany);
    for (size_t i = 0; i < howmany; i++)
        ret.push_back( pts[0]+(i*step) );

    return ret;
}

template<typename T>
std::ostream&
operator<<(std::ostream& os, const element<T>& elem)
{
    auto pts = elem.points();
    os << "Element [" << pts[0] << ", " << pts[1] << "]";
    return os;
}


