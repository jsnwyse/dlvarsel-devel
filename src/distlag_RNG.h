#ifndef _DISTLAG_RNG_
#define _DISTLAG_RNG_

#include "required_libs.h"

double distlag_inv_gauss(double mu, double lambda);

double distlag_rGIG_m_1_2(double a, double b);

double distlag_rGIG_1_2(double a, double b);

double distlag_cdf_ALD(double x, double tau, double mu, double sigma );

double distlag_pdf_ALD(double x, double tau, double mu, double sigma );

double distlag_random_truncated_normal_left( double mu, double sigma, double trunc );

double distlag_random_truncated_normal( double mu, double sigma, double trunc, int tail );

double distlag_log_pdf_mv_normal( int n, double *x, double *mu, double *lchol_prec, double scale );

#endif 