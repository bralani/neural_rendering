#include "common.h"

#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <signal.h>
#include <math.h>

#ifdef MPI_ENABLED
#  include <mpi.h>
#endif

#include "bio.h"

#include "bu/app.h"
#include "bu/bitv.h"
#include "bu/cv.h"
#include "bu/debug.h"
#include "bu/endian.h"
#include "bu/getopt.h"
#include "bu/log.h"
#include "bu/malloc.h"
#include "bu/parallel.h"
#include "bu/ptbl.h"
#include "bu/version.h"
#include "bu/vls.h"
#include "vmath.h"
#include "raytrace.h"
#include "dm.h"
#include "pkg.h"

/* private */
#include "./rtuif.h"
#include "./ext.h"
#include "brlcad_ident.h"
#include "scanline.h"
#include "rt/torch_runner.h"

/***** Variables shared with viewing model *** */
struct fb* fbp = FB_NULL;	/* Framebuffer handle */
FILE* outfp = NULL;		/* optional pixel output file */
struct icv_image* bif = NULL;

/***** end of sharing with viewing model *****/


/***** variables shared with worker() ******/
struct application APP;
int		report_progress;	/* !0 = user wants progress report */
extern int	incr_mode;		/* !0 for incremental resolution */
extern size_t	incr_nlevel;		/* number of levels */
extern render_type rt_render_type;
extern int generate_test_set;
/***** end variables shared with worker() *****/


/***** variables shared with do.c *****/
extern int	pix_start;		/* pixel to start at */
extern int	pix_end;		/* pixel to end at */
size_t		n_malloc;		/* Totals at last check */
size_t		n_free;
size_t		n_realloc;
extern int	matflag;		/* read matrix from stdin */
extern int	orientflag;		/* 1 means orientation has been set */
extern int	desiredframe;		/* frame to start at */
extern int	curframe;		/* current frame number,
					 * also shared with view.c */
extern char* outputfile;		/* name of base of output file */
extern struct icv_image* bif;
unsigned char* pixmap;
/***** end variables shared with do.c *****/


/***** variables shared with rt.c *****/
extern char* string_pix_start;	/* string spec of starting pixel */
extern char* string_pix_end;	/* string spec of ending pixel */
/***** end variables shared with rt.c *****/

/***** variables shared with view.c *****/
extern int ibackground[3];		/* integer 0..255 version */
extern int inonbackground[3];		/* integer non-background */
extern fastf_t gamma_corr;		/* gamma correction if !0 */
/***** end variables shared with view.c *****/
static struct scanline* scanline = NULL;
static size_t pwidth = 0;		/* Width of each pixel (in bytes) */
void
memory_summary(void)
{
	if (rt_verbosity & VERBOSE_STATS) {
		size_t mdelta = bu_n_malloc - n_malloc;
		size_t fdelta = bu_n_free - n_free;
		bu_log("Additional #malloc=%zu, #free=%zu, #realloc=%zu (%zu retained)\n",
			mdelta,
			fdelta,
			bu_n_realloc - n_realloc,
			mdelta - fdelta);
	}
	n_malloc = bu_n_malloc;
	n_free = bu_n_free;
	n_realloc = bu_n_realloc;
}

int fb_setup(void) {
	/* Framebuffer is desired */
	size_t xx, yy;
	int zoom;

	/* make sure width/height are set via -g/-G */
	grid_sync_dimensions(viewsize);

	/* Ask for a fb big enough to hold the image, at least 512. */
	/* This is so MGED-invoked "postage stamps" get zoomed up big
	 * enough to see.
	 */
	xx = yy = 512;
	if (xx < width || yy < height) {
		xx = width;
		yy = height;
	}

	bu_semaphore_acquire(BU_SEM_SYSCALL);
	fbp = fb_open(framebuffer, xx, yy);
	bu_semaphore_release(BU_SEM_SYSCALL);
	if (fbp == FB_NULL) {
		fprintf(stderr, "rt:  can't open frame buffer\n");
		return 12;
	}

	bu_semaphore_acquire(BU_SEM_SYSCALL);
	/* If fb came out smaller than requested, do less work */
	size_t fbwidth = (size_t)fb_getwidth(fbp);
	size_t fbheight = (size_t)fb_getheight(fbp);
	if (width > fbwidth)
		width = fbwidth;
	if (height > fbheight)
		height = fbheight;

	/* If fb is lots bigger (>= 2X), zoom up & center */
	if (width > 0 && height > 0) {
		zoom = fbwidth / width;
		if (fbheight / height < (size_t)zoom) {
			zoom = fb_getheight(fbp) / height;
		}
		(void)fb_view(fbp, width / 2, height / 2, zoom, zoom);
	}
	bu_semaphore_release(BU_SEM_SYSCALL);

#ifdef USE_OPENCL
	clt_connect_fb(fbp);
#endif
	return 0;
}

void
initialize_option_defaults(void)
{
	/* GIFT defaults */
	azimuth = 35.0;
	elevation = 25.0;

	/* 40% ambient light */
	AmbientIntensity = 0.4;

	/* 0/0/1 background */
	background[0] = background[1] = 0.0;
	background[2] = 1.0 / 255.0; /* slightly non-black */

	/* Before option processing, get default number of processors */
	npsw = bu_avail_cpus();		/* Use all that are present */
	if (npsw > MAX_PSW)
		npsw = MAX_PSW;

}

void
initialize_resources(size_t cnt, struct resource* resp, struct rt_i* rtip)
{
	if (!resp)
		return;

	/* Initialize all the per-CPU memory resources.  Number of
	 * processors can change at runtime, so initialize all.
	 */
	memset(resp, 0, sizeof(struct resource) * cnt);

	int i;
	for (i = 0; i < MAX_PSW; i++) {
		rt_init_resource(&resp[i], i, rtip);
	}
}

/**
 * Do all the actual work to run a frame.
 *
 * Returns -1 on error, 0 if OK.
 */
int
neu_do_frame(int framenumber)
{
	 struct bu_vls times = BU_VLS_INIT_ZERO;
    char framename[128] = {0};		/* File name to hold current frame */
    struct rt_i *rtip = APP.a_rt_i;
    double utime = 0.0;			/* CPU time used */
    double nutime = 0.0;		/* CPU time used, normalized by ncpu */
    double wallclock = 0.0;		/* # seconds of wall clock time */
    vect_t work, temp;
    quat_t quat;

    if (rt_verbosity & VERBOSE_FRAMENUMBER)
	bu_log("\n...................Frame %5d...................\n",
	       framenumber);

    /* Compute model RPP, etc. */
    do_prep(rtip);

    if (rt_verbosity & VERBOSE_VIEWDETAIL)
	bu_log("Tree: %zu solids in %zu regions\n", rtip->nsolids, rtip->nregions);

    if (Query_one_pixel) {
	query_optical_debug = OPTICAL_DEBUG;
	query_debug = RT_G_DEBUG;
	rt_debug = optical_debug = 0;
    }

    if (validate_raytrace(rtip) > 0)
	return -1;

    if (rt_verbosity & VERBOSE_VIEWDETAIL)
	bu_log("Model: X(%g, %g), Y(%g, %g), Z(%g, %g)\n",
	       rtip->mdl_min[X], rtip->mdl_max[X],
	       rtip->mdl_min[Y], rtip->mdl_max[Y],
	       rtip->mdl_min[Z], rtip->mdl_max[Z]);

    /*
     * Perform Grid setup.
     * This may alter cell size or width/height.
     */
    {
	int setup;
	struct bu_vls msg = BU_VLS_INIT_ZERO;

	setup = grid_setup(&msg);
	if (setup ) {
	    bu_exit(BRLCAD_ERROR, "%s\n", bu_vls_cstr(&msg));
	}

	bu_vls_free(&msg);
    }

    /* az/el 0, 0 is when screen +Z is model +X */
    VSET(work, 0, 0, 1);
    MAT3X3VEC(temp, view2model, work);
    bn_ae_vec(&azimuth, &elevation, temp);

    if (rt_verbosity & VERBOSE_VIEWDETAIL)
	bu_log("View: %g azimuth, %g elevation off of front view\n",
	       azimuth, elevation);
    quat_mat2quat(quat, model2view);

    if (rt_verbosity & VERBOSE_VIEWDETAIL) {
	bu_log("Orientation: %g, %g, %g, %g\n", V4ARGS(quat));
	bu_log("Eye_pos: %g, %g, %g\n", V3ARGS(eye_model));
	bu_log("Size: %gmm\n", viewsize);

	/**
	 * This code shows how the model2view matrix can be
	 * reconstructed using the information from the Orientation,
	 * Eye_pos, and Size messages in the rt log output.
	 @code
	 {
	 mat_t rotscale, xlate, newmat;
	 quat_t newquat;
	 bn_mat_print("model2view", model2view);
	 quat_quat2mat(rotscale, quat);
	 rotscale[15] = 0.5 * viewsize;
	 MAT_IDN(xlate);
	 MAT_DELTAS_VEC_NEG(xlate, eye_model);
	 bn_mat_mul(newmat, rotscale, xlate);
	 bn_mat_print("reconstructed m2v", newmat);
	 quat_mat2quat(newquat, newmat);
	 HPRINT("reconstructed orientation:", newquat);
	 @endcode
	 *
	 */

	bu_log("Grid: (%g, %g) mm, (%zu, %zu) pixels\n",
	       cell_width, cell_height,
	       width, height);
	bu_log("Beam: radius=%g mm, divergence=%g mm/1mm\n",
	       APP.a_rbeam, APP.a_diverge);
    }

    /* Process -b and ??? options now, for this frame */
    if (pix_start == -1) {
	pix_start = 0;
	pix_end = (int)(height * width - 1);
    }
    if (string_pix_start) {
	int xx, yy;
	register char *cp = string_pix_start;

	xx = atoi(cp);
	while (*cp >= '0' && *cp <= '9') cp++;
	while (*cp && (*cp < '0' || *cp > '9')) cp++;
	yy = atoi(cp);
	bu_log("only pixel %d %d\n", xx, yy);
	if (xx * yy >= 0) {
	    pix_start = (int)(yy * width + xx);
	    pix_end = pix_start;
	}
    }
    if (string_pix_end) {
	int xx, yy;
	register char *cp = string_pix_end;

	xx = atoi(cp);
	while (*cp >= '0' && *cp <= '9') cp++;
	while (*cp && (*cp < '0' || *cp > '9')) cp++;
	yy = atoi(cp);
	bu_log("ending pixel %d %d\n", xx, yy);
	if (xx * yy >= 0) {
	    pix_end = (int)(yy * width + xx);
	}
    }

    /* Allocate data for pixel map for rerendering of black pixels */
    if (pixmap == NULL) {
	pixmap = (unsigned char*)bu_calloc(sizeof(RGBpixel), width*height, "pixmap allocate");
    }

    /*
     * Determine output file name
     * On UNIX only, check to see if this is a "restart".
     */
    if (outputfile != (char *)0) {
	if (framenumber <= 0) {
	    snprintf(framename, 128, "%s", outputfile);
	} else {
	    snprintf(framename, 128, "%s.%d", outputfile, framenumber);
	}

	/* Ordinary case for creating output file */
	if (outfp == NULL) {
#ifndef RT_TXT_OUTPUT
	    /* FIXME: in the case of rtxray, this is wrong.  it writes
	     * out a bw image so depth should be just 1, not 3.
	     */
	    bif = icv_create(width, height, ICV_COLOR_SPACE_RGB);

	    if (bif == NULL && (outfp = fopen(framename, "w+b")) == NULL) {
		perror(framename);
		if (matflag)
		    return 0;	/* OK */
		return -1;			/* Bad */
	    }
#else
	    outfp = fopen(framename, "w");
	    if (outfp == NULL) {
		perror(framename);
		if (matflag)
		    return 0;	/* OK */
		return -1;			/* Bad */
	    }
#endif
	}

	if (rt_verbosity & VERBOSE_OUTPUTFILE)
	    bu_log("Output file is '%s' %zux%zu pixels\n", framename, width, height);
    }

    /* initialize lighting, may update pix_start */
    view_2init(&APP, framename);

#ifdef USE_OPENCL
    if (opencl_mode) {
        unsigned int mode = 0;

	mode |= CLT_COLOR;
        if (full_incr_mode)
	    mode |= CLT_ACCUM;

        clt_view_init(mode);
    }
#endif

    rtip->nshots = 0;
    rtip->nmiss_model = 0;
    rtip->nmiss_tree = 0;
    rtip->nmiss_solid = 0;
    rtip->nmiss = 0;
    rtip->nhits = 0;
    rtip->rti_nrays = 0;

    if (rt_verbosity & (VERBOSE_LIGHTINFO|VERBOSE_STATS))
	bu_log("\n");
    fflush(stdout);
    fflush(stderr);

    /*
     * Compute the image
     * It may prove desirable to do this in chunks
     */
    rt_prep_timer();

#ifdef USE_OPENCL
    if (opencl_mode) {
	clt_run(pix_start, pix_end);

	/* Reset values to full size, for next frame (if any) */
	pix_start = 0;
	pix_end = (int)(height*width - 1);
    }
    else
#endif
    if (incr_mode) {
	for (incr_level = 1; incr_level <= incr_nlevel; incr_level++) {
	    if (incr_level > 1)
		view_2init(&APP, framename);

	    do_run(0, (1<<incr_level)*(1<<incr_level)-1);
	}
    }
    else if (full_incr_mode) {
	/* Multiple frame buffer mode */
	for (full_incr_sample = 1; full_incr_sample <= full_incr_nsamples;
	    full_incr_sample++) {
	    if (full_incr_sample > 1) /* first sample was already initialized */
		view_2init(&APP, framename);
	    do_run(pix_start, pix_end);
	}
    }
    else {
	do_run(pix_start, pix_end);

	/* Reset values to full size, for next frame (if any) */
	pix_start = 0;
	pix_end = (int)(height*width - 1);
    }
    utime = rt_get_timer(&times, &wallclock);

    /*
     * End of application.  Done outside of timing section.
     * Typically, writes any remaining results out.
     */
    view_end(&APP);

    /* These results need to be normalized.  Otherwise, all we would
     * know is that a given workload takes about the same amount of
     * CPU time, regardless of the number of CPUs.
     */
    if ((size_t)npsw > 1) {
	size_t avail_cpus;
	size_t ncpus;

	avail_cpus = bu_avail_cpus();
	if ((size_t)npsw > avail_cpus) {
	    ncpus = avail_cpus;
	} else {
	    ncpus = npsw;
	}
	nutime = utime / ncpus;			/* compensate */
    } else {
	nutime = utime;
    }

    /* prevent a bogus near-zero time to prevent infinite and
     * near-infinite results without relying on IEEE floating point
     * zero comparison.
     */
    if (NEAR_ZERO(nutime, VDIVIDE_TOL)) {
	bu_log("WARNING:  Raytrace timings are likely to be meaningless\n");
	nutime = VDIVIDE_TOL;
    }

    /*
     * All done.  Display run statistics.
     */
    if (rt_verbosity & VERBOSE_STATS)
	bu_log("SHOT: %s\n", bu_vls_addr(&times));
    bu_vls_free(&times);
    memory_summary();
    if (rt_verbosity & VERBOSE_STATS) {
	bu_log("%zu solid/ray intersections: %zu hits + %zu miss\n",
	       rtip->nshots, rtip->nhits, rtip->nmiss);
	bu_log("pruned %.1f%%:  %zu model RPP, %zu dups skipped, %zu solid RPP\n",
	       rtip->nshots > 0 ? ((double)rtip->nhits*100.0)/rtip->nshots : 100.0,
	       rtip->nmiss_model, rtip->ndup, rtip->nmiss_solid);
	bu_log("Frame %2d: %10zu pixels in %9.2f sec = %12.2f pixels/sec\n",
	       framenumber,
	       width*height, nutime, ((double)(width*height))/nutime);
	bu_log("Frame %2d: %10zu rays   in %9.2f sec = %12.2f rays/sec (RTFM)\n",
	       framenumber,
	       rtip->rti_nrays, nutime, ((double)(rtip->rti_nrays))/nutime);
	bu_log("Frame %2d: %10zu rays   in %9.2f sec = %12.2f rays/CPU_sec\n",
	       framenumber,
	       rtip->rti_nrays, utime, ((double)(rtip->rti_nrays))/utime);
	bu_log("Frame %2d: %10zu rays   in %9.2f sec = %12.2f rays/sec (wallclock)\n",
	       framenumber,
	       rtip->rti_nrays,
	       wallclock, ((double)(rtip->rti_nrays))/wallclock);
    }
    if (bif != NULL) {
	icv_write(bif, framename, BU_MIME_IMAGE_AUTO);
	icv_destroy(bif);
	bif = NULL;
    }

    if (outfp != NULL) {
	(void)fclose(outfp);
	outfp = NULL;
    }

    if (OPTICAL_DEBUG&OPTICAL_DEBUG_STATS) {
	/* Print additional statistics */
	res_pr();
    }

    bu_log("\n");
    bu_free(pixmap, "pixmap allocate");
    pixmap = (unsigned char *)NULL;
    return 0;		/* OK */
}

static int
validate_raytrace(struct rt_i* rtip)
{
	if (!rtip) {
		bu_log("ERROR: No raytracing instance.\n");
		return 1;
	}
	if (rtip->nsolids <= 0) {
		bu_log("ERROR: No primitives remaining.\n");
		return 2;
	}
	if (rtip->nregions <= 0) {
		bu_log("ERROR: No regions remaining.\n");
		return 3;
	}

	return 0;
}

static void cal_rgb(struct application* ap, RGBpixel rgb)
{
	int r, g, b;
	int do_eol = 0;
	if (ap->a_user == 0) {
		/* Shot missed the model, don't dither */
		r = ibackground[0];
		g = ibackground[1];
		b = ibackground[2];
		VSETALL(ap->a_color, -1e-20);	/* background flag */
	}
	else {
		/*
		 * To prevent bad color aliasing, add some color dither.  Be
		 * certain to NOT output the background color here.  Random
		 * numbers in the range 0 to 1 are used, so that integer
		 * valued colors (e.g., from texture maps) retain their original
		 * values.
		 */
		if (!ZERO(gamma_corr)) {
			/*
			 * Perform gamma correction in floating-point space, and
			 * avoid nasty mach bands in dark areas from doing it in
			 * 0..255 space later.
			 */
			double ex = 1.0 / gamma_corr;
			r = floor(pow(ap->a_color[0], ex) * 255. +
				bn_rand0to1(ap->a_resource->re_randptr) + 0.5);
			g = floor(pow(ap->a_color[1], ex) * 255. +
				bn_rand0to1(ap->a_resource->re_randptr) + 0.5);
			b = floor(pow(ap->a_color[2], ex) * 255. +
				bn_rand0to1(ap->a_resource->re_randptr) + 0.5);
		}
		else {
			r = ap->a_color[0] * 255. + bn_rand0to1(ap->a_resource->re_randptr);
			g = ap->a_color[1] * 255. + bn_rand0to1(ap->a_resource->re_randptr);
			b = ap->a_color[2] * 255. + bn_rand0to1(ap->a_resource->re_randptr);
		}
		if (r > 255) r = 255;
		else if (r < 0) r = 0;
		if (g > 255) g = 255;
		else if (g < 0) g = 0;
		if (b > 255) b = 255;
		else if (b < 0) b = 0;
		if (r == ibackground[0] && g == ibackground[1] &&
			b == ibackground[2]) {
			r = inonbackground[0];
			g = inonbackground[1];
			b = inonbackground[2];
		}

		/* Make sure it's never perfect black */
		if (r == 0 && g == 0 && b == 0 && benchmark == 0)
			b = 1;
	}
	rgb[0] = r;
	rgb[1] = g;
	rgb[2] = b;
}

int get_hit(point_t start,vect_t dir)
{
	int cpu = 0;
	struct application a;
	struct pixel_ext pe;
	vect_t colorsum = { (fastf_t)0.0, (fastf_t)0.0, (fastf_t)0.0 };
	int samplenum = 0;
	static const double one_over_255 = 1.0 / 255.0;

	/* for stereo output */
	vect_t left_eye_delta = VINIT_ZERO;
	/* Obtain fresh copy of global application struct */
	a = APP;				/* struct copy */
	a.a_resource = &resource[cpu];

	/* Check the pixel map to determine if this image should be
	 * rendered or not.
	 */
	if (pixmap) {
		a.a_user = 1;	/* Force Shot Hit */
	}
	/* not tracing the corners of a prism by default */
	a.a_pixelext = (struct pixel_ext*)NULL;

	/* black or no pixmap, so compute the pixel(s) */

	/* LOOP BELOW IS UNROLLED ONE SAMPLE SINCE THAT'S THE COMMON CASE.
	 *
	 * XXX - If you edit the unrolled or non-unrolled section, be sure
	 * to edit the other section.
	 */
	VMOVE(a.a_ray.r_pt, start);
	VMOVE(a.a_ray.r_dir, dir);
	a.a_level = 0;		/* recursion level */
	a.a_purpose = "main ray";
	rt_shootray(&a);

	return a.a_user;
}

void set_size(int size)
{
	width = size;
	height = size;
	grid_sync_dimensions(viewsize);
	if (!orientflag)
		do_ae(azimuth, elevation);
}

void get_center(point_t center)
{
	VSETALL(center, 0.0);
	VADD2SCALE(center, APP.a_rt_i->rti_pmin, APP.a_rt_i->rti_pmax, 0.5);
}

fastf_t get_r()
{
	return APP.a_rt_i->rti_radius;
}

void set_type(render_type type)
{
	rt_render_type = type;
}


void set_generate_test_set(int generate)
{
	generate_test_set = generate;
}


int hit_sphere(const point_t center, fastf_t radius, struct xray* ray, vect_t * intersection1, vect_t * intersection2)
{
	point_t dict;
	VSUB2(dict, ray->r_pt, center);
	fastf_t a = VDOT(ray->r_dir, ray->r_dir);
	fastf_t b = 2 * VDOT(ray->r_dir, dict);
	fastf_t c = VDOT(dict, dict) - radius * radius;
	fastf_t discriminate = b * b - 4 * a * c;
	if (discriminate < 0)
		return 0;
	else
	{
        fastf_t sqrt_discriminant = sqrt(discriminate);
        fastf_t t1 = (-b - sqrt_discriminant) / (2.0 * a);
        fastf_t t2 = (-b + sqrt_discriminant) / (2.0 * a);

		vect_t copy_dir = { ray->r_pt[0], ray->r_pt[1], ray->r_pt[2] };
		vect_t dir_ray_scaled;
		VSCALE(dir_ray_scaled, ray->r_dir, t1);
		VADD2(*intersection1, copy_dir, dir_ray_scaled);

		VSCALE(dir_ray_scaled, ray->r_dir, t2);
		VADD2(*intersection2, copy_dir, dir_ray_scaled);

        return 1;
	}
	return 0;
}



void convert_to_sph_point(vect_t point, point_t origin, fastf_t r, point2d_t res)
{
	fastf_t x = point[0] - origin[0];
	fastf_t y = point[1] - origin[1];
	fastf_t z = point[2] - origin[2];
	
    fastf_t theta = acos(z / r);
    fastf_t phi = atan2(y, x);

    res[1] = theta;
    res[2] = phi;
}

// for param convert
void cert_to_sph_p(fastf_t* para, point_t pt, vect_t dir,fastf_t intersection)
{
	point_t new_point;
	point_t dis;
	VSET(new_point, 0, 0, 0);
	VSET(dis, 0, 0, 0);
	VSCALE(dis, dir, intersection);
	VADD2(new_point, pt, dis);
	point_t center;
	get_center(center);
	fastf_t x = pt[0] - center[0];
	fastf_t y = pt[1] - center[1];
	fastf_t z = pt[2] - center[2];
	para[1] = acos(z / get_r());
	para[2] = atan2(y, x);
	VUNITIZE(dir);
	para[3] = acos(dir[2]);
	para[4] = atan2(dir[1],dir[0]);
}
