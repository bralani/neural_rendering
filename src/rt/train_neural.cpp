/*                          T R A I N _ N E U R A L. C P P
 * BRL-CAD
 *
 * Copyright (c) 1985-2024 United States Government as represented by
 * the U.S. Army Research Laboratory.
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * version 2.1 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this file; see the file named COPYING for more
 * information.
 */
 /** @file rt/train_neural.cpp
  *
  *
  */

#include "rt/neu_util.h"
#include"rt/rt_trainer.h"
#include "rt/torch_runner.h"
#include "bu/getopt.h"
#include "vmath.h"
render_type rt_render_type;

struct options
{
	bool generate_dataset;
	int num_samples;
	int size;
	std::string model_path;
};

static void
get_options(int argc, char *argv[], struct options *opts)
{
    static const char *usage = "Usage: %s [-d] [-n #samples] [-m ...path_model.pt] [-s #size_img] model.g nameObject \n -d if you want to generate the dataset (1) or if you want to perform a benchmark rendering (0) \n -n #samples if you want to specify the number of rays to train the NN (default 1 million) \n -size of the image (default 256)";

    const char *argv0 = argv[0];
    const char *db = NULL;
    const char *obj = NULL;


    int c, d;
    while ((c = bu_getopt(argc, (char * const *)argv, "dn:h?")) != -1) {
	if (bu_optopt == '?')
	    c = 'h';

	switch (c) {

	    case 'n':
		if (opts)
		    opts->num_samples = (size_t)strtol(bu_optarg, NULL, 10);
		break;

	    case 's':
		if (opts)
		    opts->size = (size_t)strtol(bu_optarg, NULL, 10);
		break;

		case 'm':
			if (opts)
				opts->model_path = bu_optarg;

	    case 'd':
		if (opts)
			d = (int)strtol(bu_optarg, NULL, 10);
			opts->generate_dataset = (d == 1);
		break;
	    case '?':
	    case 'h':
		/* asking for help */
		bu_exit(EXIT_SUCCESS, usage, argv0);
	    default:
		bu_exit(EXIT_FAILURE, "ERROR: unknown option -%c\n", *bu_optarg);
	}
    }

    argv += bu_optind;

    db = argv[0];
    obj = argv[1];

    /* final sanity checks */
    if (!db || !bu_file_exists(db, NULL)) {
	bu_exit(EXIT_FAILURE, "ERROR: database %s not found\n", (db)?db:"[]");
    }
    if (!obj) {
	bu_exit(EXIT_FAILURE, "ERROR: object(s) not specified\n");
    }
}


void generate_renders_test_set(int num_renders)
{
	
	// make a file to write the test set
	FILE* file = fopen("./test_neural.txt", "w");
	fclose(file);


	// test set
	set_generate_test_set(1);
	rt_neu::render();
	for (int i = 0; i < num_renders; i++)
	{
		do_ae(rt_sample::RandomNum(-180, 180), rt_sample::RandomNum(-30, 30));
		rt_neu::render();
	}
	set_generate_test_set(0);
}

void benchmarks(int num_renders) {

	for (int i = 0; i < num_renders; i++)
	{
		do_ae(rt_sample::RandomNum(-180, 180), rt_sample::RandomNum(-30, 30));

		set_type(render_type::neural);
		rt_neu::render();

		set_type(render_type::normal);
		rt_neu::render();
	}
}


int main(int argc, char* argv[])
{
	struct options opts;
	opts.generate_dataset = true;
	opts.num_samples = 1000000;
	opts.size = 256;
	opts.model_path = "";

    char *db = NULL;
    char *ob = NULL;

	get_options(argc, argv, &opts);
    //db = argv[bu_optind];
    //ob = argv + bu_optind + 1;
	db = "C:\\Users\\m.balice\\Desktop\\rt_volume\\build\\Debug\\share\\db\\moss.g";
	ob = "all.g";

    bu_log(" db is %s\n", db);
    bu_log(" obj is %s\n", ob);

	struct rt_i* rtip = NULL;
	set_size(opts.size);
	rt_tool::init_rt(db, ob, rtip);
	do_ae(-90, 0);
	//rt_perspective = 90;
	//eye_model[0] = (fastf_t)10000.0;
	//eye_model[1] = (fastf_t)10000.0;
	//eye_model[2] = (fastf_t)10000.0;


	if (opts.generate_dataset)
	{
		bu_log("Generating dataset\n");
		
		point_t center{ 0 };
		get_center(center);

		auto ray_list_spherical = rt_sample::SampleRandomBoundingSphere(opts.num_samples);
		auto ray_list_cartesian = convert::sph_to_cert(ray_list_spherical, center, get_r());

		std::vector<int> ray_res = rt_tool::ShootSamples(ray_list_cartesian);

		// training set
		util::write_sph_json(ray_list_spherical, ray_res, "./train_neural.json");

		generate_renders_test_set(20);
	} else {
		bu_log("we\n");

		set_model_path(opts.model_path.c_str());
		bu_log("we1\n");
		//benchmarks(5);

		set_type(render_type::neural);
		bu_log("we2\n");
		rt_neu::render();
	}


	return 0;
}