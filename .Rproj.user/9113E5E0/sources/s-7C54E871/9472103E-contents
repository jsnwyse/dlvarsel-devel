#include "distlag_utils.h"

// compute a weightings matrix from basis function evaluations
void distlag_compute_normed_weight_matrix_and_deriv( int p, int ord, int l, int *varind, double *par, double *B, double *W, double *deriv_W )
{
  int i, j, k;
  double *be_ptr, *wei_ptr = W, *th_ptr = par, a, b, a0, *deriv_wei_ptr = deriv_W, *c = (double *)calloc( ord, sizeof(double) );
  for( j=0; j<p; j++ )
  {
    if( varind[j] )
    {
      be_ptr = B + j * ord * l ;
      b = 0.0;
      for( k=0; k<l; k++ )
      {
        wei_ptr[k] = 0.0;
        for( i=0; i<ord; i++ ){ wei_ptr[k] +=  th_ptr[i] * be_ptr[ i * l ]; c[i] = 0.0; } // reset c[i]
        b += wei_ptr[k];
        be_ptr += 1 ; 
      }
      b /= l ; // mean
      a = 0.0;
      be_ptr = B + j * ord * l ;
      deriv_wei_ptr = deriv_W + j * ord * l;
      for( k=0; k<l; k++ )
      {
        wei_ptr[k] = exp(wei_ptr[k] - b);
        a += wei_ptr[k];
        for( i=0; i<ord; i++ )
        {
          a0 = wei_ptr[k] * be_ptr[ i * l ];
          deriv_wei_ptr[ i*l ] = a0;
          c[i] += a0; 
        }
        be_ptr += 1;
        deriv_wei_ptr += 1;
      }
      for( i=0; i<ord; i++ ) c[i] /= a;
      deriv_wei_ptr = deriv_W + j * ord * l;
      for( k=0; k<l; k++ )
      {
        a0 = wei_ptr[k] / a;
        wei_ptr[k] = a0 ;
        for( i=0; i<ord; i++ )
        {
          deriv_wei_ptr[ i*l ] /= a;
          deriv_wei_ptr[ i*l ] -= c[i] * a0  ;
        }
        deriv_wei_ptr += 1;
      }
    }
    wei_ptr += l;
    th_ptr += ord;
  }
  free(c);
  return; 
}

void distlag_compute_weighted_reg_and_dl_linpred( int n, int p, int m, int *varind, int l, double *beta, double *W, double *X, double *XW, double *lp )
{
  int i, j, k, ii;
  double *be_ptr, *wei_ptr, *x_ptr = X, *x_w_ptr = XW, a;
  for( i=0; i<n; i++ )
  { 
    lp[i] = 0.0;
    be_ptr = beta;
    wei_ptr = W;
    for( j=0; j<p; j++ )
    { 
      for( k=0; k<m; k++ )
      {
        if( varind[j] )
        {
          a = 0.0;
          for( ii=0; ii<l; ii++ ) a += wei_ptr[ii] * x_ptr[ii]; // time direction to rely on arrangement of X on input
          x_w_ptr[k] = a;
          lp[i] += be_ptr[k] * a; 
        } 
        wei_ptr += l; // assumes copies of the basis for each variable 
      }
      x_w_ptr += m;
      be_ptr += m;
      x_ptr += l;
    }
  } 
}


void distlag_update_weighted_reg_and_dl_linpred_single_i(int i, int n, int p, int m, int *varind, int l, double *beta, 
                                                         double *W, double *X, double *XW, double *lp )
{
  int j, k, ii;
  double *be_ptr, *wei_ptr, *x_ptr = X + i * p * l, *x_w_ptr = XW + i * m * p, a;
  lp[i] = 0.0;
  be_ptr = beta;
  wei_ptr = W;
  for( j=0; j<p; j++ )
  { 
    for( k=0; k<m; k++ )
    {
      if( varind[j] )
      {
        a = 0.0;
        for( ii=0; ii<l; ii++ ) a += wei_ptr[ii] * x_ptr[ii]; // time direction to rely on arrangement of X on input
        x_w_ptr[k] = a;
        lp[i] += be_ptr[k] * a; 
      } 
      wei_ptr += l; // assumes copies of the basis for each variable 
    }
    x_w_ptr += m;
    be_ptr += m;
    x_ptr += l;
  }
}

void distlag_compute_posterior_precision_dl( int n, int p, int m, int *varind, double omega_sq, double *v, double *XW, double *prior_prec, double *posterior_prec )
{
  int i, j, k, jj, kk, idx_0, idx_1;
  double *C, *D, *x_w_ptr, *x_w_ptr_0, a;
  
  // do the lower triangle of posterior_prec
  C = posterior_prec;
  D = prior_prec;
  j = 0; jj = 0; kk = 0;
  while( j < p )
  {
    k = 0; 
    while( k < m)
    {
      idx_0 = j * m + k;
      jj = 0;  // can possibly do this better
      while( jj < p)
      {
        kk = 0;
        if( varind[j] && varind[jj] )
        {
          while( kk < m )
          {
            idx_1 = jj * m + kk;
            if( idx_1 >= idx_0 )
            {
              x_w_ptr = XW + j * m + k;
              x_w_ptr_0 = XW + jj * m + kk;
              a = 0.0;
              for(i=0; i<n; i++)
              {
                a += x_w_ptr[0] * x_w_ptr_0[0] / v[i];
                x_w_ptr += p * m;
                x_w_ptr_0 += p * m;
              }
              a /= omega_sq;
              C[kk] = a + D[kk] ;
            }
            kk += 1;
          }
        }
        C += m ;  
        D += m ;
        jj += 1;
      }
      k += 1;
    }
    j += 1;
  }
}

void distlag_compute_posterior_quantities_for_joint_sampling_sdl( int n, int q, int p, int m, int *varind, double omega_sq, double xi, double *z, double *v, double *W, double *Xw, 
                                                             double *prior_prec_stat, double *prior_prec_dl, double *posterior_prec, double *mu_term  )
{
  // computes the lower triangle of the posterior precision when all coefficients sampled jointly
  int i, j, k, jj, kk, idx_0, idx_1, *v_ptr = varind;
  double *C, *D, *x_w_ptr, *x_w_ptr_0, *w_ptr, *w_ptr_0, *mu = mu_term, a;

  // do the top left triangle for static q times q
  C = posterior_prec;
  D = prior_prec_stat;
  
  //reset 
  for( k=0; k<(q + p*m)*(q + p*m); k++ ) C[k] = 0.0;
  
  
  j = 0; jj = 0; kk = 0;
  
  for( j=0; j<q; j++ )
  {
    w_ptr = W + j * n ;
    for( i=j; i<q; i++ )
    {
      if( v_ptr[i] && v_ptr[j] ) 
      {
        w_ptr_0 = W + i * n ;
        a = 0.0 ;
        for( k=0; k<n; k++ ) a += w_ptr_0[k] * w_ptr[k] / v[k] ;
        a /= omega_sq ; 
        C[i] = a + D[i] ; // add the prior precision
      }
    }
    C += q + p * m ;
    D += q ;
    // do the mu_term
    if( v_ptr[j] )
    {
      a = 0.0;
      for( k=0; k<n; k++ ) a += w_ptr[k] * ( z[k] / v[k] - xi ) ;
      mu[j] = a / omega_sq ; 
    }
  }  
  mu += q ; // move along for the next variables

  // do the bottom right triangle for dl p times p 
  C = posterior_prec + q * ( q + p * m ) + q ; // ptr to the "(1,1)" entry of triangle
  D = prior_prec_dl ; 
  v_ptr += q ; // ptr to variable indicators for dl  
  
  for( j=0; j<p; j++ )
  {
    for( k=0; k<m; k++ )
    {
      for( jj=0; jj<p; jj++ ) 
      {
        if( v_ptr[j] && v_ptr[jj] && jj > j-1 )
        {
          idx_0 = ( jj == j ) ? k : 0 ;
          for( kk=idx_0; kk<m; kk++ )
          {
            x_w_ptr = Xw + j * m + k;
            x_w_ptr_0 = Xw + jj * m + kk;
            a = 0.0;
            for(i=0; i<n; i++)
            {
              a += x_w_ptr[0] * x_w_ptr_0[0] / v[i];
              x_w_ptr += p * m;
              x_w_ptr_0 += p * m;
            }
            a /= omega_sq;
            C[kk] = a + D[kk] ; 
          }
        }
        C += m ; 
        D += m ; 
      }
      C += q ;
      // do the mu_term here
      if( v_ptr[j] )
      {
        x_w_ptr = Xw + j * m + k;
        a = 0.0;
        for( i=0; i<n; i++ )
        {
          a += x_w_ptr[0] * ( z[i] / v[i] - xi );
          x_w_ptr += p * m ;
        }
        a /= omega_sq;
        mu[k] = a ;
      }
    }
    mu += m ;
  }
  
  // do the cross term in the bottom left pm times q-- no prior precision terms needed if indep priors
  C = posterior_prec + q ; // ptr to the "(1,1)" entry of block
  for( j=0; j<q; j++ ) // CHECK THIS
  {
    w_ptr = W + j * n ;
    for( jj=0; jj<p; jj++ )
    {
      if( varind[j] && varind[ q + jj] ) 
      {
        for( kk=0; kk<m; kk++ )
        {
          x_w_ptr = Xw + jj * m + kk;
          a = 0.0;
          for( i=0; i<n; i++ ) 
          {
            a += w_ptr[i] * x_w_ptr[0] / v[i];
            x_w_ptr += p * m ;
          }
          a /= omega_sq ; 
          C[kk] = a ;
        }
      }
      C += m ;
    }
    C += q ;
  }
  
}

void distlag_update_linear_predictor_1( int n, int q, int *varind, double *gamma, double *W, double *lp1, double *eta ) // need to add varind
{
  int i, j, k;
  double a, *w_i;
  for( i=0; i<n; i++ ) 
  {
    a = 0.0;
    w_i = W + i;
    for( j=0; j<q; j++ )
    { 
      if( varind[j] ) a += w_i[0] * gamma[j]; 
      w_i += n;  
    }
    eta[i] += a - lp1[i];
    lp1[i] = a;
  }
}

void distlag_update_linear_predictor_2( int n, int p, int m, int *varind, double *beta, double *Xw, double *lp2, double *eta )
{
  int i, j, k;
  double a, *x_w_ptr, *be_ptr;
  // recompute linear predictors
  for( i=0; i<n; i++ ) 
  {
    a = 0.0;
    x_w_ptr = Xw + i * m * p ;
    be_ptr = beta;
    for( j=0; j<p; j++ )
    {
      if( varind[j] )
      {
        for( k=0; k<m; k++ ) a += be_ptr[k] * x_w_ptr[k];
      }
      x_w_ptr += m; 
      be_ptr += m;
    }
    eta[i] += a - lp2[i];
    lp2[i] = a;
  }
}

/*void distlag_update_linear_predictor_2_single_i(int i, int n, int p, int m, int *varind, double *beta, double *Xw, double *lp2, double *eta )
{
  int j, k;
  double a, *x_w_ptr, *be_ptr;
  // recompute linear predictors
  a = 0.0;
  x_w_ptr = Xw + i * m * p ;
  be_ptr = beta;
  for( j=0; j<p; j++ )
  {
    if( varind[j] )
    {
      for( k=0; k<m; k++ ) a += be_ptr[k] * x_w_ptr[k];
    }
    x_w_ptr += m; 
    be_ptr += m;
  }
  eta[i] += a - lp2[i];
  lp2[i] = a;
}*/

void distlag_compute_deriv_WX_eta(int n, int p, int ord, int l, int *varind, double *X, double *derivW, double *derivWX_eta )
{
  int i, ii, j, k;
  double *e_ptr, *x_ptr, *deriv_wei_ptr, a;
  for( j=0; j<p; j++ )
  {
    e_ptr = derivWX_eta + j * n * ord;
    deriv_wei_ptr = derivW + j * l * ord;
    for( i=0; i<ord; i++ )
    {
      x_ptr = X + j * l ;
      if( varind[j] )
      {
        for( ii=0; ii<n; ii++ )
        {
          a = 0.0;
          for( k=0; k<l; k++ ) a += deriv_wei_ptr[k] * x_ptr[k];
          e_ptr[ii] =  a ; 
          x_ptr += p * l ;
        }
      }
      deriv_wei_ptr += l;
      e_ptr += n;
    }
  }
}

double distlag_compute_logpost_and_gradient_wrt_theta( int n, int p, int ord, int l, int wei_scheme, int *varind, int *y, double *z, double *v, double omega_sq, double xi, double tau, double *eta,
                                                       double *X, double *derivW, double *derivWX_eta, int compute_derivWX_eta, double *beta, double *theta, double *prior_mean_theta, 
                                                       double *prior_sd_theta, double *prior_rate_theta, double *grad, int compute_met_tens, double *G_mat  )
{
  int i, j, k, ii, r, s, p0, idx_0, idx_1;
  double lpost = 0.0, *x_ptr, *e_ptr, *e_ptr2, *th_ptr = theta, *grad_ptr = grad, *deriv_wei_ptr = derivW, *G_mat_ptr = G_mat, a;
  
  double *F = (double *)calloc( n, sizeof(double) ), *f = (double *)calloc( n, sizeof(double) );
  for( ii=0; ii<n; ii++ )
  {
    F[ii] = distlag_cdf_ALD( -eta[ii], tau, 0.0, 1.0 );
    f[ii] = distlag_pdf_ALD( -eta[ii], tau, 0.0, 1.0 );
    lpost +=  (1.0 - y[ii]) * log( F[ii] ) + y[ii] * log( 1.0 - F[ii] );
    //lpost += -0.5 * R_pow( z[ii] - eta[ii] - xi * v[ii] , 2.0 ) / ( omega_sq * v[ii] ) ; 
  }
  //lpost *= -0.5 / omega_sq ;
  
  // create a derivative matrix-- better to compute in a separate loop as can be used for 
  //    calculation of the expected Fisher Information
  if( compute_derivWX_eta )
  {
    for( j=0; j<p; j++ )
    {
      e_ptr = derivWX_eta + j * n * ord;
      deriv_wei_ptr = derivW + j * l * ord;
      for( i=0; i<ord; i++ )
      {
        x_ptr = X + j * l ;
        if( varind[j] )
        {
          for( ii=0; ii<n; ii++ )
          {
            a = 0.0;
            for( k=0; k<l; k++ ) a += deriv_wei_ptr[k] * x_ptr[k];
            e_ptr[ii] =  a ; 
            x_ptr += p * l ;
          }
        }
        deriv_wei_ptr += l;
        e_ptr += n;
      }
    }
  }
  
  for( j=0; j<p; j++ )
  {
    e_ptr = derivWX_eta + n * j * ord ;
    for( i=0; i<ord; i++ )
    {
      grad_ptr[i] = 0.0;
      if( varind[j] )
      {
        for( ii=0; ii<n; ii++ ) grad_ptr[i] += ( y[ii] / ( 1.0 - F[ii] ) - ( 1.0 - y[ii] ) / F[ii]  )  * f[ii] * e_ptr[ii]  ;
        //for( ii=0; ii<n; ii++ ) grad_ptr[i] += ( z[ii] - eta[ii] - xi * v[ii] ) / ( omega_sq * v[ii] ) * e_ptr[ii] ;
        //( (z[ii] - eta[ii])/v[ii] - xi ) * e_ptr[ii] ;
        grad_ptr[i] *= beta[j] ; /// omega_sq ; 
        if( wei_scheme == 0 && i == 1 )
        {
          // special gradient for the exponential prior
          grad_ptr[i] += prior_rate_theta[ j * ord + i ] + 1.0 / th_ptr[i] ; 
          grad_ptr[i] *= th_ptr[i] ; // deriv wrt to real line parameter
          lpost += dexp( -th_ptr[i], 1.0 / prior_rate_theta[ j * ord + i ], 1 ) + log( -th_ptr[i] );
        }
        else
        {
          grad_ptr[i] += - ( th_ptr[i] - prior_mean_theta[ j * ord + i ] ) / R_pow( prior_sd_theta[ j * ord + i]  , 2.0  );
          lpost += dnorm( th_ptr[i], prior_mean_theta[ j * ord + i ], prior_sd_theta[ j * ord + i] , 1 );
        }
      }
      if( compute_met_tens ) // metric tensor for manifold MALA
      {
        idx_0 = j * ord + i ;
        r = 0;
        e_ptr2 = derivWX_eta ;
        if( varind[j] )
        { 
          while( r < p )
          {
            s = 0;
            while( s < ord )
            {
              if( varind[r] )
              {
                idx_1 = r * ord + s; 
                if( idx_1 >= idx_0 )
                {
                  a = 0.0;
                  for( ii=0; ii<n; ii++ ) a += e_ptr[ii] * e_ptr2[ii] * R_pow( f[ii], 2.0 ) / ( F[ii] * ( 1.0 - F[ii] ) ) ;
                  a *= ( beta[j] * beta[r] ) ; 
                  G_mat_ptr[ idx_1 ] = a ;
                  // consider cases with transformations
                  if( wei_scheme == 0 )
                  {
                    if( i == 1 ) G_mat_ptr[ idx_1 ] *= theta[ idx_0 ];
                    if( s == 1 ) G_mat_ptr[ idx_1 ] *= theta[ idx_1 ]; 
                  }
                  // add the prior terms
                  if( j == r && i == s ) // add along the diagonal
                  {
                    if( wei_scheme == 0 && i == 1 ) 
                      G_mat_ptr[ idx_1 ] += - prior_rate_theta[ idx_0 ] * theta[ idx_0 ];
                    else 
                      G_mat_ptr[ idx_1 ] += 1.0 / R_pow( prior_sd_theta[ idx_0 ]  , 2.0  );
                  }
                }
              }
              e_ptr2 += n; // problem here in that the pointer move depends on the conditional
              s += 1;
            }
            r += 1;
          }
        }
        G_mat_ptr += p * ord ;
      }
      e_ptr += n ;
    }
    grad_ptr += ord;
    th_ptr += ord;
  }
  free(F); free(f);
  return(lpost);
}

// FUNCTIONS FOR COLLAPSING EXAPANDING MATRICES/VECTORS IN VARIABLE SELECTION SETTING

void distlag_collapse_precision_matrix( int p, int p0, int m, int *varind, double *prec )
{
  int j, jj, k, kk, dim;
  dim = p0*m;
  double *prec_reduced = (double *)calloc( dim*dim, sizeof(double) ), *prec_ptr = prec, *prec_ptr2 = prec_reduced;
  for( j=0; j<p; j++ )
  {
    if( varind[j] )
    {
      for( k=0; k<m; k++ )
      {
        for( jj=0; jj<p; jj++ )
        {
          if( varind[jj] )
          {
            for( kk=0; kk<m; kk++ ) prec_ptr2[kk] = prec_ptr[kk];
            prec_ptr2 += m ;
          }
          prec_ptr += m ;
        }
      }
    }
    else prec_ptr += ( p * m  * m ) ; 
  }
  for( k=0; k< dim*dim ; k++ ) prec[k] = prec_reduced[k];
  free( prec_reduced );
}


void distlag_collapse_joint_precision_matrix( int q, int p, int q0, int p0, int m, int *varind, double *prec )
{
  int j, jj, k, kk, dim;
  dim = q + p * m;
  double *prec_reduced = (double *)calloc( dim*dim, sizeof(double) ), *prec_ptr = prec, *prec_ptr2 = prec_reduced;
  
  for( j=0; j<q; j++ )
  {
    if( varind[j] )
    {
      // the gamma vars first
      for( jj=0; jj<q; jj++ )
      {
        if( varind[jj] ){ prec_ptr2[0] = prec_ptr[jj]; prec_ptr2 += 1; }
      }
      prec_ptr += q;
      // the beta vars second
      for( jj=0; jj<p; jj++ )
      {
        if( varind[ q + jj ] )
        {
          for( kk=0; kk<m; kk++ ) prec_ptr2[kk] = prec_ptr[kk];
          prec_ptr2 += m ;
        }
        prec_ptr += m;
      }
    }else prec_ptr += q + p * m ;// skip
  }
  // next cycle through the dl vars - remaining p * m cols
  for( j=0; j<p; j++ )
  {
    if( varind[ q + j ] )
    {
      for( k=0; k<m; k++ )
      {
        for( jj=0; jj<q; jj++ )
        {
          if( varind[jj] ){ prec_ptr2[0] = prec_ptr[jj]; prec_ptr2 += 1; }
        }
        prec_ptr += q;
        for( jj=0; jj<p; jj++ )
        {
          if( varind[ q + jj ] )
          {
            for( kk=0; kk<m; kk++ ) prec_ptr2[kk] = prec_ptr[kk];
            prec_ptr2 += m ;
          }
          prec_ptr += m ;
        }
      }
    }else prec_ptr += ( q + p * m )* m  ; 
  }
  for( k=0; k< dim*dim ; k++ ) prec[k] = prec_reduced[k];
  free( prec_reduced );
}

void distlag_collapse_vector( int p, int p0, int m, int *varind, double *mu )
{
  int j, k;
  double *mu_reduced = (double *)calloc( p0 * m, sizeof(double) ), *mu_ptr = mu, *mu_ptr2 = mu_reduced;
  for( j=0; j<p; j++ )
  {
    if( varind[j] ) 
    {
      for( k=0; k<m; k++ ) mu_ptr2[k] = mu_ptr[k];
      mu_ptr2 += m ;
    }
    mu_ptr += m;
  }
  for( k=0; k<p0*m; k++ ) mu[k] = mu_reduced[k];
  free(mu_reduced);
}

void distlag_collapse_joint_vector( int q, int p, int q0, int p0, int m, int *varind, double *mu )
{
  int j, k;
  double *mu_reduced = (double *)calloc( q0 + p0 * m, sizeof(double) ), *mu_ptr= mu, *mu_ptr2 = mu_reduced;
  for( j=0; j<q; j++ )
  {
    if( varind[j] ){ mu_ptr2[0] = mu_ptr[j]; mu_ptr2 += 1; }
  }
  mu_ptr += q;
  for( j=0; j<p; j++ )
  {
    if( varind[ q + j ] ) 
    {
      for( k=0; k<m; k++ ) mu_ptr2[k] = mu_ptr[k];
      mu_ptr2 += m ;
    }
    mu_ptr += m;
  }
  for( k=0; k<q0+p0*m; k++ ) mu[k] = mu_reduced[k];
  free(mu_reduced);
}

void distlag_expand_vector( int p, int p0, int m, int *varind, double *beta )
{
  int j, k;
  double *beta_expanded = (double *)calloc( p * m, sizeof(double) ), *be_ptr= beta, *be_ptr2 = beta_expanded;
  for( j=0; j<p; j++ )
  {
    if( varind[j] )
    {
      for( k=0; k<m; k++) be_ptr2[k] = be_ptr[k];
      be_ptr += m;
    }
    be_ptr2 += m;
  }
  for( k=0; k<p*m; k++ ) beta[k] = beta_expanded[k];
  free(beta_expanded);
}


void distlag_expand_joint_vector( int q, int p, int q0, int p0, int m, int *varind, double *beta )
{
  int j, k;
  double *beta_expanded = (double *)calloc( q + p * m, sizeof(double) ), *be_ptr= beta, *be_ptr2 = beta_expanded;
  for( j=0; j<q; j++ )
  {
    if( varind[j] ){ be_ptr2[j] = be_ptr[0]; be_ptr += 1; }
  }
  be_ptr2 += q; 
  for( j=0; j<p; j++ )
  {
    if( varind[ q + j ] )
    {
      for( k=0; k<m; k++) be_ptr2[k] = be_ptr[k];
      be_ptr += m;
    }
    be_ptr2 += m;
  }
  for( k=0; k< q + p*m; k++ ) beta[k] = beta_expanded[k];
  free(beta_expanded);
}

double distlag_compute_logprior( int p, int ord, int *varind, int wei_scheme, double *theta, double *prior_mean_theta, 
                                 double *prior_sd_theta, double *prior_rate_theta)
{
  int j, k;
  double *th_ptr = theta, *me_ptr = prior_mean_theta, *sd_ptr = prior_sd_theta, *rt_ptr = prior_rate_theta, lprior = 0.0;
  for( j=0; j<p; j++ )
  {
    if( varind[j] )
    { 
      if( wei_scheme == 0 )
      {
        lprior += dnorm( th_ptr[0], me_ptr[0], sd_ptr[0], 1 ) + dexp( th_ptr[1], 1.0/rt_ptr[1], 1 ) + log( -th_ptr[1] );
      }
      else 
      {
        for( k=0; k<ord; k++ ) lprior += dnorm( th_ptr[k], me_ptr[k], sd_ptr[k], 1 );
      }
    }
    th_ptr += ord ; 
    me_ptr += ord ; 
    sd_ptr += ord ; 
    rt_ptr += ord ;
  }
  return( lprior );
}

// SINGLE SITE UPDATES FOR THE DL PARAMETERS

double distlag_compute_loglik( int n, int *y, double tau, double *eta )
{
  int i; 
  double F, llik;
  for( i=0; i<n; i++ )
  {
    F = distlag_cdf_ALD( -eta[i], tau, 0.0, 1.0 );
    llik += (1.0 - y[i]) * log( F ) + y[i] * log( 1.0 - F );
  }
  return( llik );
}

double distlag_compute_logprior_term( int j, int ord, int wei_scheme, double *theta, double *prior_mean_theta, 
                                      double *prior_sd_theta, double *prior_rate_theta)
{
  int k;
  double *th_ptr = theta + j * ord, *me_ptr = prior_mean_theta + j * ord, 
    *sd_ptr = prior_sd_theta + j * ord, *rt_ptr = prior_rate_theta + j * ord,
    lprior = 0.0;
  if( wei_scheme == 0 )
  {
    lprior += dnorm( th_ptr[0], me_ptr[0], sd_ptr[0], 1 ) + dexp( th_ptr[1], 1.0/rt_ptr[1], 1 ) + log( -th_ptr[1] );
  }
  else
  {
    for( k=0; k<ord; k++ ) lprior += dnorm( th_ptr[k], me_ptr[k], sd_ptr[k], 1 );
  }
  return( lprior );
}

void distlag_gradient_wrt_theta_single_variable( int j, int n, int p, int ord, int l, int wei_scheme, int *varind, int *y, double *z, double *v, double omega_sq, double xi, double tau, double *eta,
                                                 double *X, double *derivW, double *derivWX_eta, int compute_derivWX_eta, double *beta, double *theta, double *prior_mean_theta, 
                                                 double *prior_sd_theta, double *prior_rate_theta, double *grad, int compute_met_tens, double *G_mat  )
{
  int i, k, ii, r, s, p0, idx_0, idx_1;
  double lpost = 0.0, *x_ptr, *e_ptr, *e_ptr2, *th_ptr = theta + j * ord, 
    *grad_ptr = grad, *deriv_wei_ptr = derivW, *G_mat_ptr = G_mat, a;
  
  double *F = (double *)calloc( n, sizeof(double) ), *f = (double *)calloc( n, sizeof(double) );
  for( ii=0; ii<n; ii++ )
  {
    F[ii] = distlag_cdf_ALD( -eta[ii], tau, 0.0, 1.0 );
    f[ii] = distlag_pdf_ALD( -eta[ii], tau, 0.0, 1.0 );
    lpost += (1.0 - y[ii]) * log( F[ii] ) + y[ii] * log( 1.0 - F[ii] );
  }
  
  // create a derivative matrix-- better to compute in a separate loop as can be used for 
  //    calculation of the expected Fisher Information
  if( compute_derivWX_eta )
  {
    e_ptr = derivWX_eta + j * n * ord;
    deriv_wei_ptr = derivW + j * l * ord;
    for( i=0; i<ord; i++ )
    {
      x_ptr = X + j * l ;
      for( ii=0; ii<n; ii++ )
      {
        a = 0.0;
        for( k=0; k<l; k++ ) a += deriv_wei_ptr[k] * x_ptr[k];
        e_ptr[ii] =  a ; 
        x_ptr += p * l ;
      }
      deriv_wei_ptr += l;
      e_ptr += n;
    }
  }
  
  e_ptr = derivWX_eta + n * j * ord ; 
  for( i=0; i<ord; i++ )
  {
    grad_ptr[i] = 0.0;
    for( ii=0; ii<n; ii++ ) grad_ptr[i] += ( y[ii] / ( 1.0 - F[ii] ) - ( 1.0 - y[ii] ) / F[ii]  )  * f[ii] * e_ptr[ii]  ;
    grad_ptr[i] *= beta[j] / omega_sq ; 
    if( wei_scheme == 0 && i == 1 )
    {
      // special gradient for the exponential prior
      grad_ptr[i] += prior_rate_theta[ j * ord + i ] + 1.0 / th_ptr[i] ; 
      grad_ptr[i] *= th_ptr[i] ; // deriv wrt to real line parameter
      lpost += dexp( -th_ptr[i], 1.0 / prior_rate_theta[ j * ord + i ], 1 ) + log( -th_ptr[i] );
    }
    else grad_ptr[i] += - ( th_ptr[i] - prior_mean_theta[ j * ord + i ] ) / R_pow( prior_sd_theta[ j * ord + i]  , 2.0  );
    
    if( compute_met_tens ) // metric tensor for manifold MALA
    {
      idx_0 = j * ord + i ;
      e_ptr2 = derivWX_eta + n * j * ord ;
      s = 0;
      while( s < ord )
      {
        idx_1 = j * ord + s; 
        if( s >= i )
        {
          a = 0.0;
          for( ii=0; ii<n; ii++ ) a += e_ptr[ii] * e_ptr2[ii] * R_pow( f[ii], 2.0 ) / ( F[ii] * ( 1.0 - F[ii] ) ) ;
          a *= ( beta[j] * beta[j] ) ; 
          G_mat_ptr[ s ] = a ;
          // consider cases with transformations
          if( wei_scheme == 0 )
          {
            if( i == 1 ) G_mat_ptr[ s ] *= theta[ idx_0 ];
            if( s == 1 ) G_mat_ptr[ s ] *= theta[ idx_1 ]; 
          }
          // add the prior terms
          if( i == s ) // add along the diagonal
          {
            if( wei_scheme == 0 && i == 1 ) 
              G_mat_ptr[ s ] += - prior_rate_theta[ idx_0 ] * theta[ idx_0 ];
            else 
              G_mat_ptr[ s ] += 1.0 / R_pow( prior_sd_theta[ idx_0 ]  , 2.0  );
          }
        }
        e_ptr2 += n;
        s += 1;
      }
      G_mat_ptr += ord ;
    }
    e_ptr += n ;
  }
  //grad_ptr += ord;
  //th_ptr += ord;
  
  free(F); free(f);
  //return(lpost);
}