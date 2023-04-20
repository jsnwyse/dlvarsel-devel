
#ifndef __MATRIX_H__
#define __MATRIX_H__

#define TRUE 1
#define FALSE 0

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <string.h>

double **Matrix_create( int n, int m ) ; 

void Matrix_destroy( int n, int m, double **X ) ;

double **Matrix_SQR_create( int n ) ;

void Matrix_SQR_destroy( int n, double **X ) ;

double Matrix_SQR_inn_prod( int n, double **X, double *y ) ;

void Matrix_set_zeros( int  n, int m, double **X ) ;

#endif
