/*                  C O M B . C
 * BRL-CAD
 *
 * Copyright (c) 2008 United States Government as represented by
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
/** @file comb.c
 *
 * The comb command.
 *
 */

#include "common.h"

#include "bio.h"
#include "cmd.h"
#include "wdb.h"
#include "ged_private.h"

int
ged_comb(struct ged *gedp, int argc, const char *argv[])
{
    register struct directory *dp;
    char *comb_name;
    register int i;
    char oper;
    static const char *usage = "comb_name <operation solid>";

    GED_CHECK_DATABASE_OPEN(gedp, BRLCAD_ERROR);
    GED_CHECK_READ_ONLY(gedp, BRLCAD_ERROR);
    GED_CHECK_ARGC_GT_0(gedp, argc, BRLCAD_ERROR);

    /* initialize result */
    bu_vls_trunc(&gedp->ged_result_str, 0);

    /* must be wanting help */
    if (argc == 1) {
	bu_vls_printf(&gedp->ged_result_str, "Usage: %s %s", argv[0], usage);
	return BRLCAD_HELP;
    }

    if (argc < 4 || MAXARGS < argc) {
	bu_vls_printf(&gedp->ged_result_str, "Usage: %s %s", argv[0], usage);
	return BRLCAD_ERROR;
    }

    /* Check for odd number of arguments */
    if (argc & 01) {
	bu_vls_printf(&gedp->ged_result_str, "error in number of args!");
	return BRLCAD_ERROR;
    }

    /* Save combination name, for use inside loop */
    comb_name = (char *)argv[1];
    if ((dp=db_lookup(gedp->ged_wdbp->dbip, comb_name, LOOKUP_QUIET)) != DIR_NULL) {
	if (!(dp->d_flags & DIR_COMB)) {
	    bu_vls_printf(&gedp->ged_result_str, "ERROR: %s is not a combination", comb_name);
	    return BRLCAD_ERROR;
	}
    }

    /* Get operation and solid name for each solid */
    for (i = 2; i < argc; i += 2) {
	if (argv[i][1] != '\0') {
	    bu_vls_printf(&gedp->ged_result_str, "bad operation: %s skip member: %s\n", argv[i], argv[i+1]);
	    continue;
	}
	oper = argv[i][0];
	if ((dp = db_lookup(gedp->ged_wdbp->dbip,  argv[i+1], LOOKUP_NOISY)) == DIR_NULL) {
	    bu_vls_printf(&gedp->ged_result_str, "skipping %s\n", argv[i+1]);
	    continue;
	}

	if (oper != WMOP_UNION && oper != WMOP_SUBTRACT && oper != WMOP_INTERSECT) {
	    bu_vls_printf(&gedp->ged_result_str, "bad operation: %c skip member: %s\n",
			  oper, dp->d_namep);
	    continue;
	}

	if (ged_combadd(gedp, dp, comb_name, 0, oper, 0, 0) == DIR_NULL) {
	    bu_vls_printf(&gedp->ged_result_str, "error in combadd");
	    return BRLCAD_ERROR;
	}
    }

    if (db_lookup(gedp->ged_wdbp->dbip, comb_name, LOOKUP_QUIET) == DIR_NULL) {
	bu_vls_printf(&gedp->ged_result_str, "Error: %s not created", comb_name);
	return BRLCAD_ERROR;
    }

    return BRLCAD_OK;
}

/*
 *			G E D _ C O M B A D D
 *
 * Add an instance of object 'objp' to combination 'name'.
 * If the combination does not exist, it is created.
 * region_flag is 1 (region), or 0 (group).
 *
 *  Preserves the GIFT semantics.
 */
struct directory *
ged_combadd(struct ged			*gedp,
	    register struct directory	*objp,
	    char			*combname,
	    int				region_flag,	/* true if adding region */
	    int				relation,	/* = UNION, SUBTRACT, INTERSECT */
	    int				ident,		/* "Region ID" */
	    int				air		/* Air code */)
{
    register struct directory *dp;
    struct rt_db_internal intern;
    struct rt_comb_internal *comb;
    union tree *tp;
    struct rt_tree_array *tree_list;
    int node_count;
    int actual_count;

    /*
     * Check to see if we have to create a new combination
     */
    if ((dp = db_lookup(gedp->ged_wdbp->dbip,  combname, LOOKUP_QUIET)) == DIR_NULL) {
	int flags;

	if (region_flag)
	    flags = DIR_REGION | DIR_COMB;
	else
	    flags = DIR_COMB;

	RT_INIT_DB_INTERNAL(&intern);
	intern.idb_major_type = DB5_MAJORTYPE_BRLCAD;
	intern.idb_type = ID_COMBINATION;
	intern.idb_meth = &rt_functab[ID_COMBINATION];

	/* Update the in-core directory */
	if ((dp = db_diradd(gedp->ged_wdbp->dbip, combname, -1, 0, flags, (genptr_t)&intern.idb_type)) == DIR_NULL)  {
	    bu_vls_printf(&gedp->ged_result_str,
			  "An error has occured while adding '%s' to the database.\n",
			  combname);
	    return DIR_NULL;
	}

	BU_GETSTRUCT(comb, rt_comb_internal);
	intern.idb_ptr = (genptr_t)comb;
	comb->magic = RT_COMB_MAGIC;
	bu_vls_init(&comb->shader);
	bu_vls_init(&comb->material);
	comb->region_id = 0;  /* This makes a comb/group by default */
	comb->tree = TREE_NULL;

	if (region_flag) {
	    comb->region_flag = 1;
	    comb->region_id = ident;
	    comb->aircode = air;
	    comb->los = gedp->ged_wdbp->wdb_los_default;
	    comb->GIFTmater = gedp->ged_wdbp->wdb_mat_default;
	    bu_vls_printf(&gedp->ged_result_str,
			  "Creating region id=%d, air=%d, GIFTmaterial=%d, los=%d\n",
			  ident, air,
			  gedp->ged_wdbp->wdb_mat_default,
			  gedp->ged_wdbp->wdb_los_default);
	} else {
	    comb->region_flag = 0;
	}
	RT_GET_TREE( tp, &rt_uniresource );
	tp->magic = RT_TREE_MAGIC;
	tp->tr_l.tl_op = OP_DB_LEAF;
	tp->tr_l.tl_name = bu_strdup( objp->d_namep );
	tp->tr_l.tl_mat = (matp_t)NULL;
	comb->tree = tp;

	if (rt_db_put_internal(dp, gedp->ged_wdbp->dbip, &intern, &rt_uniresource) < 0) {
	    bu_vls_printf(&gedp->ged_result_str, "Failed to write %s", dp->d_namep);
	    return DIR_NULL;
	}
	return dp;
    } else if (!(dp->d_flags & DIR_COMB)) {
	bu_vls_printf(&gedp->ged_result_str, "%s exists, but is not a combination\n");
	return DIR_NULL;
    }

    /* combination exists, add a new member */
    if (rt_db_get_internal(&intern, dp, gedp->ged_wdbp->dbip, (fastf_t *)NULL, &rt_uniresource) < 0) {
	bu_vls_printf(&gedp->ged_result_str, "read error, aborting\n");
	return DIR_NULL;
    }

    comb = (struct rt_comb_internal *)intern.idb_ptr;
    RT_CK_COMB(comb);

    if (region_flag && !comb->region_flag) {
	bu_vls_printf(&gedp->ged_result_str, "%s: not a region\n");
	return DIR_NULL;
    }

    if (comb->tree && db_ck_v4gift_tree(comb->tree) < 0) {
	db_non_union_push(comb->tree, &rt_uniresource);
	if (db_ck_v4gift_tree(comb->tree) < 0) {
	    bu_vls_printf(&gedp->ged_result_str, "Cannot flatten tree for editing\n");
	    rt_db_free_internal(&intern, &rt_uniresource);
	    return DIR_NULL;
	}
    }

    /* make space for an extra leaf */
    node_count = db_tree_nleaves( comb->tree ) + 1;
    tree_list = (struct rt_tree_array *)bu_calloc( node_count,
						   sizeof( struct rt_tree_array ), "tree list" );

    /* flatten tree */
    if (comb->tree) {
	actual_count = 1 + (struct rt_tree_array *)db_flatten_tree(
	    tree_list, comb->tree, OP_UNION, 1, &rt_uniresource )
	    - tree_list;
	BU_ASSERT_LONG( actual_count, ==, node_count );
	comb->tree = TREE_NULL;
    }

    /* insert new member at end */
    switch (relation) {
	case '+':
	    tree_list[node_count - 1].tl_op = OP_INTERSECT;
	    break;
	case '-':
	    tree_list[node_count - 1].tl_op = OP_SUBTRACT;
	    break;
	default:
	    bu_vls_printf(&gedp->ged_result_str, "unrecognized relation (assume UNION)\n");
	case 'u':
	    tree_list[node_count - 1].tl_op = OP_UNION;
	    break;
    }

    /* make new leaf node, and insert at end of list */
    RT_GET_TREE( tp, &rt_uniresource );
    tree_list[node_count-1].tl_tree = tp;
    tp->tr_l.magic = RT_TREE_MAGIC;
    tp->tr_l.tl_op = OP_DB_LEAF;
    tp->tr_l.tl_name = bu_strdup( objp->d_namep );
    tp->tr_l.tl_mat = (matp_t)NULL;

    /* rebuild the tree */
    comb->tree = (union tree *)db_mkgift_tree( tree_list, node_count, &rt_uniresource );

    /* and finally, write it out */
    if (rt_db_put_internal(dp, gedp->ged_wdbp->dbip, &intern, &rt_uniresource) < 0) {
	bu_vls_printf(&gedp->ged_result_str, "Failed to write %s", dp->d_namep);
	return DIR_NULL;
    }

    bu_free((char *)tree_list, "combadd: tree_list");

    return (dp);
}


/*
 * Local Variables:
 * mode: C
 * tab-width: 8
 * indent-tabs-mode: t
 * c-file-style: "stroustrup"
 * End:
 * ex: shiftwidth=4 tabstop=8
 */