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
//#include "brlcad_ident.h"
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
extern unsigned char* pixmap;
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

int get_hit(point_t start,vect_t dir)
{
	int cpu = 0;
	struct application a;

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

fastf_t get_r(void)
{
	return APP.a_rt_i->rti_radius;
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
