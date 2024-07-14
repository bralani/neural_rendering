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
	double azim;
	double elev;
	std::string model_path;
	int neural_render;
};

static void
get_options(int argc, char *argv[], struct options *opts)
{
    static const char *usage = "Usage: %s [-d] [-n #samples] [-m ...path_model.pt] [-s #size_img] [-a azimuth] [-e elevation] [-v] model.g nameObject \n -d if you want to generate the dataset (1) or if you want to perform a benchmark rendering (0) \n -n #samples if you want to specify the number of rays to train the NN (default 1 million) \n -size of the image (default 256) \n -a azimuth of the camera \n -e elevation of the camera \n -v to perform a neural rendering";

    const char *argv0 = argv[0];
    const char *db = NULL;
    const char *obj = NULL;


    bu_optind = 1;

    int c, d;
    while ((c = bu_getopt(argc, (char * const *)argv, "dn:m:s:a:e:vh?")) != -1) {
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

		case 'a':
			if (opts)
				opts->azim = (double)strtol(bu_optarg, NULL, 10);
		break;

		case 'e':
			if (opts)
				opts->elev = (double)strtol(bu_optarg, NULL, 10);
		break;

		case 'v':
			if (opts)
				opts->neural_render = 1;

		case 'm':
			if (opts)
				opts->model_path = bu_optarg;
		break;

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
	bu_log(usage);
	bu_exit(EXIT_FAILURE, "ERROR: database %s not found\n", (db)?db:"[]");
    }
    if (!obj) {
	bu_log(usage);
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


int main(int argc, char* argv[])
{
	
    char *db = NULL;
    char *ob = NULL;

	struct options opts;
	db = "C:\\Users\\m.balice\\Desktop\\rt_volume\\build\\Debug\\share\\db\\moss.g";	// path to the database
	ob = "all.g";								// name of the object to render
	opts.generate_dataset = true;				// set true if you want to generate the dataset, false if you want to perform a benchmark rendering
	
	// options for the dataset generation (useful only if generate_dataset=true)
	opts.num_samples = 1000000;					// number of samples to generate
	
	// options for the rendering (useful only if generate_dataset=false)
	opts.neural_render = 0;						// set to 1 if you want to perform a neural rendering, 0 if you want to perform a normal rendering	
	opts.size = 256;							// size of the image (useful only if generate_dataset=false)
	opts.model_path = "C:\\Users\\m.balice\\Desktop\\Rendernn\\models\\model_sph1.pt";	// path to the model (already trained)
	opts.azim = 0;								// azimuth of the camera
	opts.elev = 0;								// elevation of the camera


	//get_options(argc, argv, &opts);
    //db = argv[bu_optind];
    //ob = argv[bu_optind + 1];

    bu_log(" db is %s\n", db);
    bu_log(" obj is %s\n", ob);

	struct rt_i* rtip = NULL;
	set_size(opts.size);
	rt_tool::init_rt(db, ob, rtip);
	do_ae(opts.azim, opts.elev);
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
		set_model_path(opts.model_path.c_str());

		if (opts.neural_render) {
			set_type(render_type::neural);
		}
		else {
			set_type(render_type::normal);
		}

		rt_neu::render();
	}


	return 0;
}