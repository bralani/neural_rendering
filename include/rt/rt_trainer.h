/*                    R T _ T R A I N E R . H
 * BRL-CAD
 *
 * Copyright (c) 1993-2024 United States Government as represented by
 * the U.S. Army Research Laboratory.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * version 2.1 as published by the Free Software Foundation.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this file; see the file named COPYING for more
 * information.
 */
#ifndef RT_RT_TRAINER_H
#define RT_RT_TRAINER_H

#include<vmath.h>
#include<vector>
#include<array>
#include<dm.h>
#include "nlohmann/json.hpp"
#include "rt/neu_util.h"
namespace rt_tool
{
	extern "C"
	{
		extern int get_hit(point_t start, vect_t dir);
	}
	void init_rt(const char* database_name, const char* object_name, struct rt_i* rtip);
	std::vector<int> ShootSamples(const RayParam& ray_list);
}

namespace rt_sample
{
	// generate a random number between begin and end
	double RandomNum(double begin, double end);
	// generate two points randomly in a sphere
	RayParam SampleRandomBoundingSphere(size_t num);
}

namespace rt_neu
{
	// begin neural rendering
	void render();
}

#endif // !RT_RT_TRAINER_H