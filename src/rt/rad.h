/*
 *  GIFT/SRIM Radfile record format
 *
 *  Five kinds of records, each 72 (18*4) bytes long consisting of
 *  a four character identifier and 17 single precision (32bits) floats.
 *
 * 	Header		"head"	followed by two additional records
 *	Firing		"fire"	one per ray fired (if it intersected)
 *	Reflection	"refl"	one per intersection point
 *	Escape		"escp"	one per escaped ray (after last intersection)
 *	End
 */
union radrec {
	/* Header Record */
	struct {
		char	head[4];
		float	id;		/* idrun */
		float	miview;		/* - iview */
		float	cx, cy, cz;	/* aimpoint: tcenter(1,2,3) */
		float	back;		/* backoff */
		float	e, a;		/* elevation, azimuth (deg) */
		float	vert, horz;	/* height, width of Em plane */
		float	nvert, nhorz;	/* num rays each direction */
		float	maxrfl;		/* maximum number of reflections */
		float	iview;
		float	pad1, pad2, pad3;
	} h;

	/* Firing Record */
	struct {
		char	head[4];
		float	pad1;		/* always zero */
		float	irf;		/* number of ray segments */
		float	ox, oy, oz;	/* firing point: xb(1,2,3) */
		float	h, v;		/* Em plane coordinates */
		float	ih, iv;		/* Em plane cell number */
		float	pad2[8];
	} f;

	/* Reflection Record */
	struct {
		char	head[4];
		float	packedid;	/* never zero */
		float	sight;		/* 1 = line of sight, else 0 */
		float	ix, iy, iz;	/* intersection: xb(1,2,3) */
		float	nx, ny, nz;	/* normal: wb(1,2,3) */
		float	px, py, pz;	/* principal: pdir(1,2,3) */
		float	rc1, rc2;	/* curvature */
		float	dfirst;		/* path length */
		float	ireg;		/* region */
		float	isol;		/* solid */
		float	isurf;		/* surface */
	} r;

	/* Escape record */
	struct {
		char	head[4];
		float	zero;		/* always zero */
		float	sight;		/* -2 yes, -3 no */
		float	pad1[6];
		float	dx, dy, dz;	/* dir cosines of Em vector */
		float	pad2[6];
	} e;

	/* End record */
	struct {
		char	head[4];
		float	zero[2];	/* always zero */
		float	pad[15];
	} end;

	/* raw data */
	struct {
		char	head[4];
		float	pad[17];
	} p;
};