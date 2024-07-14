#include "common.h"

#include"rt/rt_trainer.h"
#include"rt/application.h"
#include "raytrace.h"
#include "rt/torch_runner.h"
#include "vmath.h"
#include "icv.h"
#include "bu/cv.h"
#include "dm.h"
#include "bv/plot3.h"
#include "photonmap.h"
#include "scanline.h"
#include <fstream>
#include <string>
#include <random>
#include "./ext.h"
#include "rt/neu_util.h"
/***** Variables shared with viewing model *** */


struct soltab* kut_soltab = NULL;
int ambSlow = 0;
int ambSamples = 0;
double ambRadius = 0.0;
double ambOffset = 0.0;
vect_t ambient_color = { 1, 1, 1 };	/* Ambient white light */
int ibackground[3] = { 0 };		/* integer 0..255 version */
int inonbackground[3] = { 0 };		/* integer non-background */
fastf_t gamma_corr = 0.0;		/* gamma correction if !0 */


namespace convert
{

	RayParamSph cert_to_sph(RayParam& datas, point_t origin, fastf_t r)
	{
		RayParamSph res;
		res.reserve(datas.size());
		fastf_t elevation(0.0);
		fastf_t azimuth(0.0);
		fastf_t pelevation(0.0);
		fastf_t pazimuth(0.0);
		for (auto& data : datas)
		{
			fastf_t x = data.first[0] - origin[0];
			fastf_t y = data.first[1] - origin[1];
			fastf_t z = data.first[2] - origin[2];
			elevation = acos(z / r);
			azimuth = atan2(y , x);
			if (pow(data.second[0], 2) + pow(data.second[1], 2) + pow(data.second[2], 2) != 1)
			{
				fastf_t square_sum = pow(pow(data.second[0], 2) + pow(data.second[1], 2) + pow(data.second[2], 2), 0.5);
				data.second[0] /= square_sum;
				data.second[1] /= square_sum;
				data.second[2] /= square_sum;
			}
			pelevation = acos(data.second[2]);
			azimuth = atan2(data.second[1] , data.second[0]);
			res.push_back(std::make_pair(std::make_pair(elevation, azimuth), std::make_pair(pelevation, azimuth)));
		}
		return res;
	}

	
	RayParam sph_to_cert(RayParam& datas, point_t origin, fastf_t r)
	{
		RayParam res;
		res.reserve(datas.size());
		fastf_t square_sum(0);
		
		std::vector<fastf_t> p;
		std::vector<fastf_t> d;
		for (auto& data : datas)
		{
			p.clear();
			d.clear();

			fastf_t x1 = r * sin(data.first[0]) * cos(data.first[1]) + origin[0];
			fastf_t y1 = r * sin(data.first[0]) * sin(data.first[1]) + origin[1];
			fastf_t z1 = r * cos(data.first[0]) + origin[2];
			
			fastf_t x2 = r * sin(data.second[0]) * cos(data.second[1]) + origin[0];
			fastf_t y2 = r * sin(data.second[0]) * sin(data.second[1]) + origin[1];
			fastf_t z2 = r * cos(data.second[0]) + origin[2];
			
			p.push_back(x1);
			p.push_back(y1);
			p.push_back(z1);

			d.push_back(x2 - x1);
			d.push_back(y2 - y1);
			d.push_back(z2 - z1);
			square_sum = pow(pow(d[0], 2) + pow(d[1], 2) + pow(d[2], 2), 0.5);
			d[0] /= square_sum;
			d[1] /= square_sum;
			d[2] /= square_sum;

			res.push_back(std::make_pair(p, d));
		}

		return res;
	}

}


namespace util
{
	void create_plot(const char* db_name, const RayParam& rays, const char* plot_name)
	{
		std::ofstream command_file;
		command_file.open("C:\\works\\soc\\rainy\\brlcad\\neu_build\\bin\\script\\" + std::string(db_name) + ".mged");
		if (command_file.is_open())
		{
			for (int i = 1; i < rays.size() + 1; ++i)
			{
				command_file << "in " << std::string(plot_name) << "point" << i << ".s " << "sph " << rays[i - 1].first[0] << " ";
				command_file << rays[i - 1].first[1] << " ";
				command_file << rays[i - 1].first[2] << " ";
				command_file << "1" << std::endl;
			}
			command_file << "r " << std::string(plot_name) << " ";
			for (int i = 1; i < rays.size() + 1; ++i)
			{
				command_file << "u " << std::string(plot_name) << "point" << i << ".s ";
			}
			command_file << "B " << std::string(plot_name);
		}
		else
		{
			bu_log("fail to open/create file");
		}
		command_file.close();
		std::ofstream draw_file;
		draw_file.open("C:\\works\\soc\\rainy\\brlcad\\neu_build\\bin\\script\\" + std::string(db_name) + ".bat");
		if (draw_file.is_open())
		{
			draw_file << "@echo off" << std::endl;
			draw_file << "cd \"C:\\works\\soc\\rainy\\brlcad\\neu_build\\bin\"" << std::endl;
			draw_file << "mged.exe " << "\"C:\\works\\soc\\rainy\\brlcad\\build\\share\\db\\" + std::string(db_name) << ".g\"" << "<" << "\"C:\\works\\soc\\rainy\\brlcad\\neu_build\\bin\\script\\" << std::string(db_name) << ".mged\"" << std::endl;
			draw_file << "archer.exe " << "\"C:\\works\\soc\\rainy\\brlcad\\build\\share\\db\\" + std::string(db_name) << ".g\"";
		}
		else
		{
			bu_log("fail to open/create file");
		}
	}

	void write_json(const RayParam& para, const Rayres& res, const char* path)
	{
		json js_res = json::array();
		for (int i = 0; i < para.size(); i++)
		{
			json single_res = json::object();
			single_res["point"] = para[i].first;
			single_res["dir"] = para[i].second;
			single_res["rgb"] = res[i];
			js_res.push_back(single_res);
		}
		std::ofstream o(path);
		o << js_res;
		o.close();
	}
	
	void write_sph_json(const RayParam& para, const std::vector<int>& res, const char* path)
	{
		json js_res = json::array();
		for (int i = 0; i < para.size(); i++)
		{
			json single_res = json::object();
			single_res["point_sph"] = para[i].first;
			single_res["dir_sph"] = para[i].second;
			single_res["label"] = res[i];
			js_res.push_back(single_res);
		}
		std::ofstream o(path);
		o << js_res;
		o.close();
	}
}

namespace rt_sample
{
	double RandomNum(double a, double b)
	{
		std::random_device rd;
		std::mt19937 gen(rd());
		std::uniform_real_distribution<> dis(a, b);
		return dis(gen);
	}
	
	RayParam SampleRandomBoundingSphere(size_t num)
	{
		RayParam  res;
		fastf_t square_sum(0);
		std::vector<fastf_t> p;
		std::vector<fastf_t> d;

		for (int i = 0; i < num; ++i)
		{
			p.clear();
			d.clear();
			p.push_back(RandomNum(0, 3.1415926));
			p.push_back(RandomNum(-3.1415926, 3.1415926));
			d.push_back(RandomNum(0, 3.1415926));
			d.push_back(RandomNum(-3.1415926, 3.1415926));
			res.push_back(std::make_pair(p, d));
		}
		return res;
	}
}

namespace rt_tool
{
	void init_rt(const char* database_name, const char* object_name, struct rt_i* rtip)
	{
		char idbuf[2048] = { 0 };	/* First ID record info */

		initialize_option_defaults();
		/* global application context */
		RT_APPLICATION_INIT(&APP);
		/* Before option processing, do RTUIF app-specific init */
		application_init();
		APP.a_logoverlap = ((void (*)(struct application*, const struct partition*, const struct bu_ptbl*, const struct partition*))0);

		if ((rtip = rt_dirbuild(database_name, idbuf, sizeof(idbuf))) == RTI_NULL) {
			bu_exit(2, "Error building dir in train neural!\n");
		}
		APP.a_rt_i = rtip;
		/* If user gave no sizing info at all, use 512 as default */
		if (width <= 0 && cell_width <= 0)
			width = 512;
		if (height <= 0 && cell_height <= 0)
			height = 512;
		/* Copy values from command line options into rtip */
		APP.a_rt_i->rti_space_partition = space_partition;
		APP.a_rt_i->useair = use_air;
		APP.a_rt_i->rti_save_overlaps = save_overlaps;
		if (rt_dist_tol > 0) {
			APP.a_rt_i->rti_tol.dist = rt_dist_tol;
			APP.a_rt_i->rti_tol.dist_sq = rt_dist_tol * rt_dist_tol;
		}
		if (rt_perp_tol > 0) {
			APP.a_rt_i->rti_tol.perp = rt_perp_tol;
			APP.a_rt_i->rti_tol.para = 1 - rt_perp_tol;
		}
		if (rt_verbosity & VERBOSE_TOLERANCE)
			rt_pr_tol(&APP.a_rt_i->rti_tol);

		/* before view_init */
		if (outputfile && BU_STR_EQUAL(outputfile, "-"))
			outputfile = (char*)0;
		grid_sync_dimensions(viewsize);

		/* per-CPU preparation */
		initialize_resources(sizeof(resource) / sizeof(struct resource), resource, rtip);

		def_tree(APP.a_rt_i);
		const char* trees[] = { "all.g" };
		rt_gettrees(APP.a_rt_i, 1, trees, (size_t)npsw);

		view_init(&APP, (char*)database_name, (char*)object_name, outputfile != (char*)0, framebuffer != (char*)0);

		do_ae(azimuth, elevation);
		int fb_status = fb_setup();
		if (fb_status) {
			fb_log("fail to open fb");
			return;
		}
		do_prep(rtip);
		view_2init(&APP, "");
	}
	std::vector<int> ShootSamples(const RayParam& ray_list) {
		std::vector<int> res;
		point_t point{ 0 };
		vect_t dir{ 0 };
		for (auto const& ray : ray_list)
		{
			VSET(point, ray.first[0], ray.first[1], ray.first[2]);
			VSET(dir, ray.second[0], ray.second[1], ray.second[2]);
			
			res.push_back(rt_tool::get_hit(point, dir));
		}
		return res;
	}
}

namespace rt_neu
{
	void render()
	{
		do_frame(curframe);
		if (fbp != FB_NULL) {
			fb_close(fbp);
		}
	}
}

