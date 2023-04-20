/*
Methods for inverting square positive definite matrices by using 
LAPACK routines.

Author:	Jason Wyse,
			Discipline of Statistics,
			School of Computer Science and Statistics,
			Trinity College Dublin, 
			Dublin 2, Ireland
			email: wyseja@tcd.ie	
			
Last modified: Sat 04 Nov 2017 17:32:20 GMT 
*/

#include "Cholesky.h"

struct cholesky *cholesky_create( int n )
{
	struct cholesky *Chol = (struct cholesky *) malloc( sizeof(struct cholesky) );
	Chol->n = n; //size
	Chol->L = (double *)calloc( n*n, sizeof(double) ) ; 
	Chol->Inv = (double *)calloc( n*n, sizeof(double) ) ;
	Chol->logdet = -DBL_MAX ;
	return( Chol );
}

void cholesky_destroy( struct cholesky *Chol )
{	
	free( Chol->L );
	free( Chol->Inv );
	free( Chol );
	return;	
}

//Compute the Cholesky decomposition (column major ordering)
void cholesky_decomp( int n, double *X, struct cholesky *Chol, int inverse )
{
	int i, j, err;
	double sum;
	//struct cholesky *Chol = cholesky_create( n );
	// external Fortran call to Lapack routines
	F77_CALL(dpotrf)( "L", &n, X, &n, &err );
	if( err != 0 ){ Chol->err = TRUE; return; }
	// copy the output into L
	double *L = Chol->L;
	for( i=0; i<n*n; i++ ) L[i] = X[i];
	//determinant
	sum = 0.0;
	L = Chol->L; //reset
	for( i=0; i<n; i++ ){ sum += log( L[ 0 ] ) ; L += (n + 1); } // are limits on loop correct?
	Chol->logdet = 2.0 * sum ;
	if( inverse )
	{
	  F77_CALL(dpotri)("L", &n, X, &n, &err );
	  for( i=0; i<n*n; i++ ) Chol->Inv[i] = X[i];
	  for( j=1; j<n; j++)
	  {
	    for( i=0; i<j; i++ ) Chol->Inv[ n*j + i ] = Chol->Inv[ n*i + j ] ;
	  }
	}
	Chol->err = FALSE ;
  return ;
} 

// need a separate function for the inversion function
void cholesky_inv_solve( struct cholesky *Chol, double *b )
{
  int n = Chol->n, nrhs = 1, err;
  double *L = Chol->L ;
  F77_CALL(dpotrs)( "L", &n, &nrhs, L, &n, b, &n, &err ) ;
  Chol->err = err ;
  return;
}

//Solve the equation Lx = b or L^T x = b for x where L is the Cholesky (lower) triangle
// useful for generation of multivariate normal random variates where 
// b is a vector of independent N(0,1)'s
void cholesky_bf_solve( struct cholesky *Chol, double *b, int transpose )
{
	int n=Chol->n, nrhs = 1, err;
	double *L = Chol->L ;
	if( transpose ) 
	  F77_CALL(dtrtrs)( "L", "T", "N", &n, &nrhs, L, &n, b, &n, &err );
	else
	  F77_CALL(dtrtrs)( "L", "N", "N", &n, &nrhs, L, &n, b, &n, &err );
	return;
	Chol->err = err ; 
}


