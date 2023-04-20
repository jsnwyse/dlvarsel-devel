
#ifndef __CHOLESKY_H__
#define __CHOLESKY_H__

#include "required_libs.h"
//#include "Matrix.h"

struct cholesky
{
	int err;
	int n;
	double *L;
	double *Inv;
	double logdet;
};

struct cholesky *cholesky_create( int n );

void cholesky_destroy( struct cholesky *Chol );

void cholesky_decomp( int n, double *X, struct cholesky *Chol, int inverse );

void cholesky_inv_solve( struct cholesky *Chol, double *b );
  
void cholesky_bf_solve( struct cholesky *Chol, double *b, int transpose ) ;

#endif
