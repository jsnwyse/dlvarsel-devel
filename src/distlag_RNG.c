#include "distlag_RNG.h"

// simulate inverse Gaussian distribution
double distlag_inv_gauss(double mu, double lambda)
{
  double y, x, a = 0.5 * mu / lambda ;
  y = rnorm(0.0,1.0); y *= y ; y *= mu ;// chi-sq 1
  x = 4.0 * lambda * y +  y * y  ;
  x = sqrt( x ) ;
  x *= -a ;
  x += mu + a * y ;
  a = mu / (mu + x) ;
  if( runif(0.0,1.0) < a ) return x ; else return mu * mu / x ;
}

// simulate Generalized Inverse Gaussian with rate -1/2
double distlag_rGIG_m_1_2(double a, double b)
{
  double  mu = sqrt( b / a );
  // determine lambda and mu from a and b and call rng for IG dist
  return( distlag_inv_gauss( mu, b) ) ;  
}

// simulate Generalized Inverse Gaussian with rate 1/2
double distlag_rGIG_1_2(double a, double b)
{
  // use the GIG reciprocal property
  return 1.0 / distlag_rGIG_m_1_2(b, a) ;
}

double distlag_cdf_ALD(double x, double tau, double mu, double sigma )
{
  double z = (x - mu)/sigma ;
  if( z < 0.0 )
  {
    return  tau * exp( z * (1.0-tau) ) ; 
  }else{
    return tau + (1.0 - tau) * (1.0 - exp( - tau * z ) ) ;
  }
}

double distlag_pdf_ALD(double x, double tau, double mu, double sigma )
{
  double z = (x - mu)/sigma, l, rho ;
  if( z < 0 ) rho = z * ( tau - 1.0 ) ;  else rho = z * tau ;
  l = log(tau) + log(1.0 - tau) - rho ;
  return( exp(l) );
}

double distlag_random_truncated_normal_left( double mu, double sigma, double trunc )
{
  double a, z ;
  trunc = (trunc - mu) / sigma ;
  a = 0.5 * ( trunc + sqrt( trunc * trunc + 4.0) );
  while(1)
  {
    z = trunc - log( runif(0.0,1.0) ) / a ; 
    if( log( runif(0.0,1.0) )  < - 0.5 * (z-a) * (z-a) ) return( mu + z * sigma );
  }
}

double distlag_random_truncated_normal( double mu, double sigma, double trunc, int tail )
{
  if( tail > 0 ) 
    return( distlag_random_truncated_normal_left( mu, sigma, trunc) ) ; 
  else
    return( -distlag_random_truncated_normal_left( -mu, sigma, -trunc) ) ;
}

double distlag_log_pdf_mv_normal( int n, double *x, double *mu, double *lchol_prec, double scale )
{
  int i,j;
  double *L, *LL = lchol_prec, ld = 0.0, a, b=0.0, logscale= log(scale);
  //lower cholesky given
  L = lchol_prec;
  for( i=0; i<n; i++ )
  {
    a = 0.0;
    ld += log( L[i] ) - logscale ;
    for( j=i; j<n; j++ )
    {
      a += L[j] * (x[j] - mu[j]) / scale ;
    }
    b += a * a ;
    L += n ;
  }
  b *= -0.5;
  b -=  n * M_LN_SQRT_2PI ;
  b += ld ;
  return(b);
}




