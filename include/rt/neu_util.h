/*                    R T _ N E U _ U T I L  . H
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
#ifndef RT_NEU_UTIL_H
#define RT_NEU_UTIL_H
#include<vector>
#include<vmath.h>
#include "nlohmann/json.hpp"
#include<dm.h>
#include "rt/torch_runner.h"
extern "C"
{
	extern int curframe;		/* from main.c */
	extern double airdensity;	/* from opt.c */
	extern double haze[3];		/* from opt.c */
	extern int do_kut_plane;        /* from opt.c */
	extern plane_t kut_plane;       /* from opt.c */
	extern fastf_t rt_perspective;
	extern point_t eye_model;
	extern struct fb* fbp;
	extern fastf_t	rt_dist_tol;		/* Value for rti_tol.dist */
	extern fastf_t	rt_perp_tol;		/* Value for rti_tol.perp */
	/***** variables from neural.c *****/
	extern int view_init(struct application* ap, char* file, char* obj, int minus_o, int minus_F);
	extern int ao_raymiss(register struct application* ap);
	extern int ao_rayhit(register struct application* ap,
		struct partition* PartHeadp,
		struct seg* UNUSED(segp));
	extern void
		ambientOcclusion(struct application* ap, struct partition* pp);
	extern int
		colorview(struct application* ap, struct partition* PartHeadp, struct seg* finished_segs);
	extern void
		application_init(void);
	/***** variables from do.c *****/
	extern void
		do_prep(struct rt_i* rtip);
	extern int
		do_frame(int framenumber);
	extern void
		do_ae(double azim, double elev);
	/***** variables from neural.c *****/
	extern int fb_setup(void);
	extern void
		initialize_option_defaults(void);
	extern void
		initialize_resources(size_t cnt, struct resource* resp, struct rt_i* rtip);
	extern void
		view_2init(struct application* ap, char* UNUSED(framename));
	extern void set_size(int size);
	extern void get_center(point_t center);
	extern fastf_t get_r();
	extern void set_type(render_type type);
	extern void set_generate_test_set(int generate);
	extern int hit_sphere(const point_t center, fastf_t radius, struct xray* ray, vect_t * intersection1, vect_t * intersection2);
}

const std::string global_model_path;

using RayParam = std::vector<std::pair< std::vector<fastf_t>, std::vector<fastf_t>>>;
using RayParamSph = std::vector<std::pair<std::pair<fastf_t, fastf_t>, std::pair<fastf_t, fastf_t>>>;
using RGBdata = std::array<int, 3>;
using Rayres = std::vector<RGBdata>;
using json = nlohmann::json;


namespace convert
{
	// Converting Cartesian coordinates to spherical coordinates
	RayParamSph cert_to_sph(RayParam& datas, point_t origin, fastf_t r);
	// Converting spherical coordinates to Cartesian coordinates
	RayParam sph_to_cert(RayParam& datas, point_t origin, fastf_t r);

	// convert data from RGBpixel to RGBData
	RGBdata pix_to_rgb(const RGBpixel data);
}


namespace util
{
	// write a mged script
	void create_plot(const char* db_name, const RayParam& rays, const char* plot_name);
	// write reslut to a json file
	void write_json(const RayParam& para, const Rayres& res, const char* path);
	// write spherical result to a json file
	void write_sph_json(const RayParam& para, const std::vector<int>& res, const char* path);
}
#endif // !RT_RT_TRAINER_H