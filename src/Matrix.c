
#include "Matrix.h"

double **Matrix_create( int n, int m )
{
	double **X = calloc( n, sizeof(double*) );
	
	int k;
	for( k=0; k<n; k++ )
		X[k] = calloc( m, sizeof(double) );
		
	return( X );
}

void Matrix_destroy( int n, int m, double **X )
{
	int k;
	
	for( k=0; k<n; k++ ) free( X[k] );
	free(X);
	
	return;
}


double **Matrix_SQR_create( int n )
{
	return( Matrix_create( n, n ) );
}

void Matrix_SQR_destroy( int n, double **X )
{
	Matrix_destroy( n, n, X );	
	return;
}

//Get the inner product y^T X y of a square matrix
double Matrix_SQR_inn_prod( int n, double **X, double *y )
{
	int i, j;
	double sum , inn_prod = 0. ;
	
	for( i=0; i<n; i++ )
	{
		sum = 0.;
		for( j=0; j<n; j++ ) sum += X[i][j] * y[j] ;
		inn_prod += y[i] * sum ; 	
	}
	
	return( inn_prod );
}

void Matrix_set_zeros( int  n, int m, double **X )
{
	int i, j;
	for( i=0; i<n; i++ )
	{
		for( j=0; j<n; j++ ) X[i][j] = 0.;
	}
}



