/*
 *			M A T C H . C
 *
 *  Author:		Gary S. Moss
 *  
 *  Source -
 *	SECAD/VLD Computing Consortium, Bldg 394
 *	The U. S. Army Ballistic Research Laboratory
 *	Aberdeen Proving Ground, Maryland  21005-5066
 *  
 *  Copyright Notice -
 *	This software is Copyright (C) 1986-2004 by the United States Army.
 *	All rights reserved.
 */
#ifndef lint
static const char RCSid[] = "@(#)$Header$ (BRL)";
#endif

#ifdef HAVE_CONFIG_H
# include "config.h"
#endif



#include <stdio.h>
#ifdef USE_STRING_H
#include <string.h>
#else
#include <strings.h>
#endif
#include <setjmp.h>

#include "machine.h"
#include "externs.h"

#define NUL	'\0'

/*	m a t c h ( )
	if string matches pattern, return 1, else return 0
	special characters:
		*	Matches any string including the null string.
		?	Matches any single character.
		[...]	Matches any one of the characters enclosed.
		[!..]	Matches any character NOT enclosed.
		-	May be used inside brackets to specify range
			(i.e. str[1-58] matches str1, str2, ... str5, str8)
		\	Escapes special characters.
 */
int
match(register char *pattern, register char *string)
{
	do
		{
		switch( pattern[0] )
		{
		case '*': /* Match any string including null string.	*/
			if( pattern[1] == NUL || string[0] == NUL )
				return	1;
			while( string[0] != NUL )
				{
				if( match( &pattern[1], string ) )
					return	1;
				++string;
				}
			return	0;
		case '?': /* Match any character.			*/
			break;
		case '[': /* Match one of the characters in brackets
				unless first is a '!', then match
				any character not inside brackets.
			   */
			{ register char	*rgtBracket;
			  static int	negation;

			++pattern; /* Skip over left bracket.		*/
			/* Find matching right bracket.			*/
			if( (rgtBracket = strchr( pattern, ']' )) == NULL )
				{
				(void) fprintf( stderr, "Unmatched '['." );
				return	0;
				}
			/* Check for negation operator.			*/
			if( pattern[0] == '!' )
				{
				++pattern;
				negation = 1;
				}
			else	{
				negation = 0;
				}	
			/* Traverse pattern inside brackets.		*/
			for(	;
				pattern < rgtBracket
			     &&	pattern[0] != string[0];
				++pattern
				)
				{
				if(	pattern[ 0] == '-'
				    &&	pattern[-1] != '\\'
					)
					{
					if(	pattern[-1] <= string[0]
					    &&	pattern[-1] != '['
					    &&	pattern[ 1] >= string[0]
					    &&	pattern[ 1] != ']'
					)
						break;
					}
				}
			if( pattern == rgtBracket )
				{
				if( ! negation )
					{
					return	0;
					}
				}
			else
				{
				if( negation )
					{
					return	0;
					}
				}
			pattern = rgtBracket; /* Skip to right bracket.	*/
			break;
			}
		case '\\': /* Escape special character.			*/
			++pattern;
			/* WARNING: falls through to default case.	*/
		default:  /* Compare characters.			*/
			if( pattern[0] != string[0] )
				return	0;
		}
		++pattern;
		++string;
		}
	while( pattern[0] != NUL && string[0]  != NUL );
	if( (pattern[0] == NUL || pattern[0] == '*' ) && string[0]  == NUL )
		{
		return	1;
		}
	else
		{
		return	0;
		}
	}