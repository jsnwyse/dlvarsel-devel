/*---------------------------------------------------------------------------*/
/* define macros for GCC attributes                                          */

#ifndef __GIG_RVG_H__
#define __GIG_RVG_H__

#include "required_libs.h"

/*---------------------------------------------------------------------------*/

//SEXP rgig(SEXP sexp_n, SEXP sexp_lambda, SEXP sexp_chi, SEXP sexp_psi);
/*---------------------------------------------------------------------------*/
/* Draw sample from GIG distribution.                                        */
/* Wrapper for do_rgig():                                                    */
/*   GetRNGstate(); do_rgig(...); PutRNGstate();                             */
/*---------------------------------------------------------------------------*/

double do_rgig(double lambda, double chi, double psi);

double _gig_mode(double lambda, double omega);

/* Type 1 */
double _rgig_ROU_noshift (double lambda, double lambda_old, double omega, double alpha);

/* Type 4 */
double _rgig_newapproach1 (double lambda, double lambda_old, double omega, double alpha);

/* Type 8 */
double _rgig_ROU_shift_alt (double lambda, double lambda_old,  double omega, double alpha);

double _unur_bessel_k_nuasympt (double x, double nu, int islog, int expon_scaled);

/*---------------------------------------------------------------------------*/
/* Draw sample from GIG distribution                                         */
/* without calling GetRNGstate() ... PutRNGstate()                           */
/*---------------------------------------------------------------------------*/

//SEXP dgig(SEXP sexp_x, SEXP sexp_lambda, SEXP sexp_chi, SEXP sexp_psi, SEXP sexp_logvalue);
/*---------------------------------------------------------------------------*/
/* evaluate pdf of GIG distribution                                          */
/*---------------------------------------------------------------------------*/

#endif