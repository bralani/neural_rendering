/*                 RationalUniformCurve.cpp
 * BRL-CAD
 *
 * Copyright (c) 1994-2010 United States Government as represented by
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
/** @file RationalUniformCurve.cpp
 *
 * Routines to convert STEP "RationalUniformCurve" to BRL-CAD BREP
 * structures.
 *
 */

#include "STEPWrapper.h"
#include "Factory.h"

#include "RationalUniformCurve.h"
#include "CartesianPoint.h"

#define CLASSNAME "RationalUniformCurve"
#define ENTITYNAME "Rational_Uniform_Curve"
string RationalUniformCurve::entityname = Factory::RegisterClass(ENTITYNAME,(FactoryMethod)RationalUniformCurve::Create);

RationalUniformCurve::RationalUniformCurve() {
	step = NULL;
	id = 0;
}

RationalUniformCurve::RationalUniformCurve(STEPWrapper *sw,int step_id) {
	step = sw;
	id = step_id;
}

RationalUniformCurve::~RationalUniformCurve() {
}

bool
RationalUniformCurve::Load(STEPWrapper *sw,SCLP23(Application_instance) *sse) {
	step=sw;
	id = sse->STEPfile_id;

	// load base class attributes
	if ( !RationalBSplineCurve::Load(sw,sse) ) {
		std::cout << CLASSNAME << ":Error loading base class ::RationalBSplineCurve." << std::endl;
		return false;
	}

	return true;
}

void
RationalUniformCurve::Print(int level) {
	TAB(level); std::cout << CLASSNAME << ":" << name << "(";
	std::cout << "ID:" << STEPid() << ")" << std::endl;

	TAB(level); std::cout << "Inherited:" << std::endl;
	RationalBSplineCurve::Print(level+1);
}
STEPEntity *
RationalUniformCurve::Create(STEPWrapper *sw,SCLP23(Application_instance) *sse){
	Factory::OBJECTS::iterator i;
	if ((i = Factory::FindObject(sse->STEPfile_id)) == Factory::objects.end()) {
		RationalUniformCurve *object = new RationalUniformCurve(sw,sse->STEPfile_id);

		Factory::AddObject(object);

		if (!object->Load(sw,sse)) {
			std::cerr << CLASSNAME << ":Error loading class in ::Create() method." << std::endl;
			delete object;
			return NULL;
		}
		return static_cast<STEPEntity *>(object);
	} else {
		return (*i).second;
	}
}

// Local Variables:
// tab-width: 8
// mode: C++
// c-basic-offset: 4
// indent-tabs-mode: t
// c-file-style: "stroustrup"
// End:
// ex: shiftwidth=4 tabstop=8