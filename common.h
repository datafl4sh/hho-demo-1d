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

struct run_parameters
{
    int         degree;
    int         num_elements;
    int         eval_per_elem;
    char *      filename;
    bool        draw;
};

