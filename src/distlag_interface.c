
#include "required_libs.h"
#include "Cholesky.h"
#include "distlag_RNG.h"
#include "distlag_utils.h"
#include "GIGrvg.h"

static void distlag_rGIG_half(int *n, double  *x, double *a, double *b);
static void distlag_rtALD(int *n, double *x, double *tau, double *mu, double *sigma, int *tail);
static void distlag_MCMC_bspl(int *n, int *y, int *q, double *W, int *p, double *X, int *l, int *m, double *tau, double *prec_pr_sl,
                              double *prec_pr_sl_sp, double *basis_eval,
                              double *gamma_samp, double *eta_samp, double *beta_samp, double *theta_samp, int *var_samp, double *loglik,
                              double *prprob_samp,
                              int *nsamples, int *burn, int *thin,
                              int *static_vars, int *dl_vars, int *weight_scheme, double *init_theta, 
                              double *prior_mean_theta, double *prior_sd_theta, double *prior_rate_theta,
                              int *neal_ord, double *sd_theta_prop, int *adapt_interval, double *acc_rt_theta, int *langevin_samp,
                              int *dl_vars_varsel, int *varind, double *sd_varsel, double *varind_prior_probs,
                              double *hprior_varind, int *num_miss, int *idx_miss, double *miss_samp, int *verbose);


static void distlag_rGIG_half(int *n, double  *x, double *a, double *b)
{
  GetRNGstate();
  int k;
  for( k=0; k<*n; k++ ) x[k] = distlag_rGIG_1_2( *a, *b );
  PutRNGstate();
  return;
}

static void distlag_MCMC_bspl(int *n, int *y, int *q, double *W, int *p, double *X, 
                              int *l, int *m, double *tau, double *prec_pr_sl,
                              double *prec_pr_sl_sp, double *basis_eval,
                              double *gamma_samp, double *eta_samp, double *beta_samp, double *theta_samp, int *var_samp, double *loglik,
                              double *prprob_samp,
                              int *nsamples, int *burn, int *thin, 
                              int *static_vars, int *dl_vars, int *weight_scheme, double *init_theta, 
                              double *prior_mean_theta, double *prior_sd_theta, double *prior_rate_theta,
                              int *neal_ord, double *sd_theta_prop, int *adapt_interval, double *acc_rt_theta, int *langevin_samp,
                              int *dl_vars_varsel, int *varind, double *sd_varsel, double *varind_prior_probs,
                              double *hprior_varind, int *num_miss, int *idx_miss, double *miss_samp, int *verbose)
{
  // declaration and defn of constants
  int i, j, k, ii, jj, kk, samp, idx_0, idx_1, dim, p0, q0, err, nrhs=1, iter, niter = *burn + (*nsamples) * (*thin),
    *proposed = (int *)R_alloc( *neal_ord, sizeof(int) ), *accepted = (int *)R_alloc( *neal_ord, sizeof(int) ),
    *varind_prop = (int *)R_alloc( *q + *p, sizeof(int) ), *vi_ptr, varsel_prop = 0, varsel_acc = 0, *var_s = var_samp;
  double *gamma_s = gamma_samp, *eta_s = eta_samp, *beta_s = beta_samp, *theta_s = theta_samp, *loglik_s = loglik, *prpr_s = prprob_samp,
    *v = (double *)R_alloc( *n, sizeof(double) ), 
    *z = (double *)R_alloc( *n, sizeof(double)) , 
    *ng = (double *)R_alloc( *q, sizeof(double)),
    *nb = (double *)R_alloc( (*p)*(*m), sizeof(double)),
    *nl = (double *)R_alloc( (*p)*(*neal_ord), sizeof(double)),
    *njoint = (double *)R_alloc( *q + (*p)*(*m), sizeof(double) ),
    *gamma = (double *)R_alloc( *q, sizeof(double)), 
    *beta = (double *)R_alloc( (*p) * (*m), sizeof(double) ),
    *lp1 = (double *)R_alloc( *n, sizeof(double)),
    *lp2 = (double *)R_alloc( *n, sizeof(double)),
    *lp2_prop = (double *)R_alloc(*n, sizeof(double)),
    *eta = (double *)R_alloc( *n, sizeof(double)),
    *eta_prop = (double *)R_alloc( *n, sizeof(double)),
    *wei = (double *)R_alloc( (*l) * (*m) * (*p), sizeof(double) ), *wei_ptr, // m 'basis' functions
    *wei_prop = (double *)R_alloc( (*l) * (*m) * (*p), sizeof(double) ), *wei_prop_ptr,
    *X_w = (double *)R_alloc( (*n) * (*m) * (*p), sizeof(double) ), *x_w_ptr, *x_w_ptr_0, 
    *X_w_prop = (double *)R_alloc( (*n) * (*m) * (*p), sizeof(double) ),
    *theta = (double *)R_alloc( (*p) * (*neal_ord), sizeof(double) ),
    *theta_prop = (double *)R_alloc( (*p) * (*neal_ord), sizeof(double) ),
    *x_ptr, *be_ptr, *th_ptr, *th_prop_ptr,// post processed t-s
    omega_sq = 2.0 / (*tau * (1 - *tau) ) , 
    xi = 0.5 * ( 1.0 - 2.0 * (*tau)) * omega_sq, 
    xi_sq = xi * xi,  
    psi = xi_sq / omega_sq + 2.0 , 
    chi, a, b, g0, g1, prior_var_ind_prob, logpost_curr, loglik_curr, logpost_prop, loglik_prop, 
    logprior_curr, logprior_prop, logprop_term,
    lproposal_ratio,  log_acc_ratio, lvn_step = sd_theta_prop[0],
    //*mu_gamma = (double *)R_alloc( *q, sizeof(double) ), 
    //*mu_beta = (double *)R_alloc( (*p) * (*m), sizeof(double)),
    *mu_joint = (double *)R_alloc( *q + (*p) * (*m), sizeof(double) ),
    //*post_prec_gamma = (double *)R_alloc( (*q) * (*q), sizeof(double) ), 
    //*post_prec_beta = (double *)R_alloc( (*p) * (*m) * (*p) * (*m), sizeof(double) ),
    *post_prec_joint = (double *)R_alloc( (*q + (*p) * (*m)) * (*q + (*p) * (*m)), sizeof(double) ),
    *transf_theta = (double *)R_alloc( (*p) * (*neal_ord), sizeof(double) ),
    *transf_prop_theta = (double *)R_alloc( (*p) * (*neal_ord), sizeof(double) ),
    *grad_curr = (double *)R_alloc( (*neal_ord) * (*p), sizeof(double) ),
    *grad_prop = (double *)R_alloc( (*neal_ord) * (*p), sizeof(double) ),
    *grad_ptr,
    *deriv_wei_curr = (double *)R_alloc( (*l) * (*neal_ord) * (*p), sizeof(double) ),
    *deriv_wei_prop = (double *)R_alloc( (*l) * (*neal_ord) * (*p), sizeof(double) ),
    *deriv_wei_ptr,
    *deriv_deriv_wx_curr = (double *)R_alloc( (*n) * (*neal_ord) * (*p), sizeof(double) ),
    *deriv_deriv_wx_prop = (double *)R_alloc( (*n) * (*neal_ord) * (*p), sizeof(double) ),
    *G_mat_curr = (double *)R_alloc( (*neal_ord) * (*p) * (*neal_ord) * (*p), sizeof(double) ),
    *G_mat_prop = (double *)R_alloc( (*neal_ord) * (*p) * (*neal_ord) * (*p), sizeof(double) ),
    *mu_lvn = (double *)R_alloc( (*neal_ord) * (*p), sizeof(double) ), *mu_ptr,
    *miss_holder = (double *)R_alloc( *num_miss, sizeof(double) ),
    *C, *D, *w_i, *w_j, *x_i, *x_j, *L, ld, ar_phi = 0.95, sigma_imp = 0.5;  
    
    GetRNGstate();
  
    for( i=0; i<*neal_ord; i++ ) {accepted[i] = 0; proposed[i] = 0;}
    for( i=0; i<(*q + (*p) * (*m)) * (*q + (*p) * (*m)); i++ ) post_prec_joint[i] = 0.0;
    for( i=0; i<(*neal_ord) * (*p) * (*neal_ord) * (*p) ; i++ ){ G_mat_curr[i] = 0.0; G_mat_prop[i] = 0.0; }
      
    // INITIALIZE
    if( *static_vars )
    {
      for( i=0; i<*q; i++ ) gamma[i] = rnorm(0.0,0.1); // do something better with this?
      for( i=0; i<*n; i++ ) 
      {
        a = 0.0;
        w_i = W + i;
        for( k=0; k<*q; k++ ){ a += w_i[0] * gamma[k]; w_i += *n;  }
        lp1[i] = a;
      }
    }
    else
    {
      for( i=0; i<*q; i++ ) gamma[i] = 0.0; 
      for( i=0; i<*n; i++ ) lp1[i] = 0.0; 
    }
    
    q0 = 0;
    for( k=0; k<*q; k++ ) q0 += varind[k];
    vi_ptr = varind + *q;
    p0 = 0;
    for( k=0; k<*p; k++ ) p0 += vi_ptr[k];
    
    if( *dl_vars && p0 > 0 )
    {
      be_ptr = beta;
      for( j=0; j<*p; j++ )
      {
        if( vi_ptr[j] )
        {
          for( k=0; k<*m; k++ ) be_ptr[k] = rnorm(0.0,0.1);
        }else{
          for( k=0; k<*m; k++ ) be_ptr[k] = 0.0;
        }
        be_ptr += *m;
      }
      if( *weight_scheme < 2 )
      {
        for( k=0; k<(*p)*(*neal_ord); k++ ) theta[k] = init_theta[k];
        // compute the normalized weight matrix for all p distributed lag variables
        distlag_compute_normed_weight_matrix_and_deriv( *p, *neal_ord, *l, varind + *q, theta, basis_eval, wei, deriv_wei_curr );
      }else{
        // *weight_scheme == 2 -- copy over the evaluated basis functions
        wei_ptr = wei;
        be_ptr = basis_eval;
        for( j=0; j<*p; j++ )
        {
          for( k=0; k<*m; k++ )
          {
            for( ii=0; ii<*l; ii++ ) wei_ptr[ii] = be_ptr[ii];
            wei_ptr += *l;
            be_ptr += *l;
          }
        }
      }
      distlag_compute_weighted_reg_and_dl_linpred( *n, *p, *m, varind + *q, *l, beta, wei, X, X_w, lp2 );
    }
    else
    {
      for( i=0; i<(*p)*(*m); i++ ) beta[i] = 0.0 ; 
      for( i=0; i<*n; i++ ) lp2[i] = 0.0; 
    }
    
    // intialize v, eta
    for( i=0; i<*n; i++ )
    {
      v[i] = rexp(1.0); 
      eta[i] = lp1[i] + lp2[i] ; // maybe other random effects
    }
    
    // do an initial run to compute the eta deriv
    if( *weight_scheme < 2 && *dl_vars )
    {
      for( i=0; i<*n; i++ ) z[i] = distlag_random_truncated_normal( eta[i] + xi * v[i], sqrt( omega_sq * v[i] ), 0.0, 2*y[i]-1 ) ;
      logpost_curr =  distlag_compute_logpost_and_gradient_wrt_theta( *n, *p, *neal_ord, *l, *weight_scheme, varind + *q, y, z, v, omega_sq, xi, *tau, eta, X, 
                                                                      deriv_wei_curr, deriv_deriv_wx_curr, 1, beta, theta, prior_mean_theta, 
                                                                      prior_sd_theta, prior_rate_theta, grad_curr, 0, G_mat_curr );
    }
     
    for( iter=0; iter<niter; iter++ )
    {  
      // check to see if interrupted
      R_CheckUserInterrupt();
      
      // UPDATE FOR LATENT PARAMETER SET z values
      for( i=0; i<*n; i++ )
      {
        // UPDATE FOR LATENT PARAMETER SET z values
        z[i] = distlag_random_truncated_normal( eta[i] + xi * v[i], sqrt( omega_sq * v[i] ), 0.0, 2*y[i]-1 ) ;
        // UPDATE FOR LATENT PARAMETER SET nu values
        chi = (z[i] - eta[i]) * (z[i] - eta[i]) / omega_sq ; 
        v[i] = do_rgig( 0.5, chi, psi );
      }
      
      // combine gamma and beta to do Gibbs sampling of all coefficients
      
      // do a joint update of gamma and beta
      distlag_compute_posterior_quantities_for_joint_sampling_sdl( *n, *q, *p, *m, varind, omega_sq, xi, z, v, W, X_w, 
                                                                   prec_pr_sl, prec_pr_sl_sp, post_prec_joint, mu_joint );

      // need to do collapse here
      if( q0 < *q || p0 < *p ) 
      {
        distlag_collapse_joint_precision_matrix( *q, *p, q0, p0, *m, varind, post_prec_joint ); 
        distlag_collapse_joint_vector( *q, *p, q0, p0, *m, varind, mu_joint );
      }
      dim = q0 + p0 * (*m) ; 
      // Cholesky factor in post_prec_joint
      F77_CALL(dpotrf)( "L", &dim, post_prec_joint, &dim, &err ); 
      if( err!= 0 ) error("failure to compute Cholesky factor: joint sampling");
      // solve for mean of full conditional of gamma 
      F77_CALL(dpotrs)( "L", &dim, &nrhs, post_prec_joint, &dim, mu_joint, &dim, &err ) ;
      if( err!= 0 ) error("failure to solve Cholesky system: system solve in mean for joint sampling");
      // now sample
      for( i=0; i<dim; i++ ) njoint[i] = rnorm( 0.0, 1.0 ) ;
      F77_CALL(dtrtrs)( "L", "T", "N", &dim, &nrhs, post_prec_joint, &dim, njoint, &dim, &err );
      if( err!= 0 ) error("failure to solve Cholesky system: posterior joint sampling");
      if( q0 < *q || p0 < *p )
      {
        // expand the vectors
        distlag_expand_joint_vector( *q, *p, q0, p0, *m, varind, mu_joint );
        distlag_expand_joint_vector( *q, *p, q0, p0, *m, varind, njoint );
      }
      for( i=0; i<*q; i++ ) gamma[i] = mu_joint[i] + njoint[i] ;
      for( i=0; i<(*p)*(*m); i++ ) beta[i] = mu_joint[*q + i] + njoint[*q + i];
      if( *static_vars ) distlag_update_linear_predictor_1( *n, *q, varind, gamma, W, lp1, eta );
      if( *dl_vars ) distlag_update_linear_predictor_2( *n, *p, *m, varind + *q, beta, X_w, lp2, eta );
      
      if( *num_miss > 0 && *weight_scheme == 2 )
      {
        for( k=0; k<*num_miss; k++ )
        {
          i = idx_miss[k]; // individual 
          j = idx_miss[*num_miss + k]; // variable
          ii = idx_miss[*num_miss * 2 + k]; // time index within variable
          // get the form of the mean first
          a = z[i] - xi * v[i] - eta[i];
          // compute part to be added back
          wei_ptr = W + j * (*m) * (*l) + ii ; // points to relevant time index
          be_ptr = beta + j * (*m) ;
          b = 0.0;
          for( kk=0; kk<*m; kk++ ){ b += be_ptr[kk] * wei_ptr[0]; wei_ptr += *l; }
          x_ptr = X + i * (*p) * (*l) + j * (*l);
          a += b * x_ptr[ii] ; 
          g1 = b * b / (omega_sq * v[i]);
          // make a in to mean
          g0 = a * b / (omega_sq * v[i]);
          if(ii > 0){ g0 += ar_phi * x_ptr[ii-1] / (sigma_imp * sigma_imp) ; g1 += 1/(sigma_imp * sigma_imp); }
          if(ii < *l-1){ g0 += ar_phi * x_ptr[ii+1] / (sigma_imp * sigma_imp) ; g1 += ar_phi * ar_phi/(sigma_imp * sigma_imp);}
          //if( iter <  10 ) Rprintf( "\t mean =  %lf, var = %lf \n", g0/g1, 1.0/g1 );
          //if( iter <  10 ) Rprintf( "\t x_ptr[before] =  %lf\n", x_ptr[ii]);
          x_ptr[ii] = rnorm( g0/g1, sqrt(1.0/g1) ) ; 
          //if( iter <  10 ) Rprintf( "\t x_ptr[after] =  %lf\n", x_ptr[ii]);
          //if( iter <  10 ) Rprintf( "\t mean = %lf  sd = %lf\n ", a / b, sqrt( omega_sq * v[i] / (b*b) ) );
          //eta[i] -= lp2[i] ;
          distlag_update_weighted_reg_and_dl_linpred_single_i( i, *n, *p, *m, varind + *q, *l, beta, wei, X, X_w, lp2 );
          eta[i] = lp1[i] + lp2[i] ;
          miss_holder[k] = x_ptr[ii];
          //if( iter <  10 ) Rprintf( "\t a = %lf  b = %lf ", a, b);
          //if( iter <  10 ) Rprintf( "\n\t %lf   \n", x_ptr[ii]);
        }
        //if( iter <  10 ){ Rprintf( "\n  %lf ", x_ptr[kk]); Rprintf( "\n   ", x_ptr[kk]);
      }
      
      p0 = 0; q0 = 0;
      for( j=0; j<*q; j++ ) q0 += varind[ j ];
      if( *dl_vars ) for( j=0; j<*p; j++ ) p0 += varind[ *q + j ];
      
      if( *dl_vars )
      {
        // MALA / SIMPLIFIED MANIFOLD MALA UPDATE FOR NEALMON WEIGHTINGS OR B-SPLINE COEFFICIENTS
        if( *weight_scheme < 2 && p0 > 0 )
        {
          proposed[0] += 1;
          
          vi_ptr = varind + *q ;
          
          // compute log posterior and the gradient at current point (transformed version)
          logpost_curr =  distlag_compute_logpost_and_gradient_wrt_theta( *n, *p, *neal_ord, *l, *weight_scheme, varind + *q , y, z, v, omega_sq, xi, *tau, eta, X, 
                                                                          deriv_wei_curr, deriv_deriv_wx_curr, 0, beta, theta, prior_mean_theta, 
                                                                          prior_sd_theta, prior_rate_theta, grad_curr,1-*langevin_samp, G_mat_curr  );
          
          // transformed version of theta
          for( k=0; k<(*p)*(*neal_ord); k++ )
          {
            transf_theta[k] = theta[k];
          }

          if( *weight_scheme == 0 )
          {
            for( j=0; j<*p; j++ ) transf_theta[ j*(*neal_ord) + 1 ] = log( - theta[ j*(*neal_ord) + 1 ] );
          }
          
          // take a step along the gradient
          lproposal_ratio = 0.0;
          
          if( *langevin_samp )
          {
            mu_ptr = mu_lvn; th_ptr = transf_theta; th_prop_ptr = transf_prop_theta; grad_ptr = grad_curr;
            for( j=0; j<*p; j++ )
            {
              if( vi_ptr[j] )
              {
                for( k=0; k<*neal_ord; k++ )
                {
                  mu_ptr[k] = th_ptr[k] + 0.5 * lvn_step * lvn_step * grad_ptr[k];
                  th_prop_ptr[k] = mu_ptr[k] + rnorm(0.0, lvn_step );
                  lproposal_ratio -= dnorm( th_prop_ptr[k], mu_ptr[k], lvn_step, 1);
                }
              }
              else
              {
                for( k=0; k<*neal_ord; k++ ) th_prop_ptr[k] = 0.0;
              }
              mu_ptr += *neal_ord;
              th_ptr += *neal_ord;
              th_prop_ptr += *neal_ord;
              grad_ptr += *neal_ord;
            }
            for( k=0; k<(*p)*(*neal_ord); k++ ) theta_prop[k] = transf_prop_theta[k];
          }
          else
          {
            // sampling with the metric tensor
            if( p0 < *p ) distlag_collapse_precision_matrix( *p, p0, *neal_ord, varind + *q, G_mat_curr );
            // Cholesky factor of expected information
            dim = (p0) * (*neal_ord);
            F77_CALL(dpotrf)( "L", &dim, G_mat_curr, &dim, &err );
            if( err!= 0 ) error("failure to compute Cholesky factor: metric sampling at current point");
            // get the conditional mean
            for( k=0; k<(*p)*(*neal_ord); k++ ) mu_lvn[k] = grad_curr[k];
            if( p0 < *p ) distlag_collapse_vector( *p, p0, *neal_ord, varind + *q, mu_lvn );
            F77_CALL(dpotrs)( "L", &dim, &nrhs, G_mat_curr, &dim, mu_lvn, &dim, &err ) ;
            if( err!= 0 ) error("failure to solve Cholesky system: inverse on gradient metric sampling");
            if( p0 < *p ) distlag_collapse_vector( *p, p0, *neal_ord, varind + *q, transf_theta );
            for( k=0; k<(p0)*(*neal_ord); k++ ){ mu_lvn[k] *= 0.5 * lvn_step * lvn_step ; mu_lvn[k] += transf_theta[k]; }
            if( p0 < *p ) distlag_expand_vector( *p, p0, *neal_ord, varind + *q, transf_theta );
            for( k=0; k<(p0)*(*neal_ord); k++ ){ nl[k] = rnorm(0.0, lvn_step ); }
            F77_CALL(dtrtrs)( "L", "T", "N", &dim, &nrhs, G_mat_curr, &dim, nl, &dim, &err );
            if( err!= 0 ) error("failure to solve Cholesky system: system solve for generation current");
            for( k=0; k<(p0)*(*neal_ord); k++ ){ transf_prop_theta[k] = mu_lvn[k] + nl[k] ; }
            lproposal_ratio -= distlag_log_pdf_mv_normal( dim, transf_prop_theta, mu_lvn, G_mat_curr, lvn_step ); 
            if( p0 < *p ) distlag_expand_vector( *p, p0, *neal_ord, varind + *q, transf_prop_theta );
            for( k=0; k<(*p)*(*neal_ord); k++ ) theta_prop[k] = transf_prop_theta[k];
          }
          
          if( *weight_scheme == 0 )
          {
            for( j=0; j<*p; j++ ) theta_prop[ j*(*neal_ord) + 1] = - exp( theta_prop[ j*(*neal_ord) + 1] ); 
          }
          
          // need to compute the backward gradient
          distlag_compute_normed_weight_matrix_and_deriv( *p, *neal_ord, *l, varind + *q, theta_prop, basis_eval, wei_prop, deriv_wei_prop );
          // compute the updated linear predictor and reg mats
          distlag_compute_weighted_reg_and_dl_linpred( *n, *p, *m, varind + *q, *l, beta, wei_prop, X, X_w_prop, lp2_prop );
          // compute the updated eta
          for( i=0; i<*n; i++ ) eta_prop[i] = lp1[i] + lp2_prop[i];
          // compute log-posterior and gradient at proposed point
          logpost_prop = distlag_compute_logpost_and_gradient_wrt_theta( *n, *p, *neal_ord, *l, *weight_scheme, varind + *q, y, z, v, omega_sq, xi, *tau, eta_prop, X, 
                                                                         deriv_wei_prop, deriv_deriv_wx_prop, 1, beta, theta_prop, prior_mean_theta, 
                                                                         prior_sd_theta, prior_rate_theta, grad_prop, 1-*langevin_samp, G_mat_prop );
          
          if(*langevin_samp )
          {
            mu_ptr = mu_lvn; th_ptr = transf_theta; th_prop_ptr = transf_prop_theta; grad_ptr = grad_prop;
            for( j=0; j<*p; j++ )
            {
              if( vi_ptr[j] )
              {
                for( k=0; k<*neal_ord; k++ )
                {
                  mu_ptr[k] = th_prop_ptr[k] + 0.5 * lvn_step * lvn_step * grad_ptr[k];
                  lproposal_ratio += dnorm( th_ptr[k], mu_ptr[k], lvn_step, 1);
                }
              }
              mu_ptr += *neal_ord;
              th_ptr += *neal_ord;
              th_prop_ptr += *neal_ord;
              grad_ptr += *neal_ord;
            }
          }
          else
          {
            if( p0 < *p ) distlag_collapse_precision_matrix( *p, p0, *neal_ord, varind + *q, G_mat_prop );
            dim = (p0) * (*neal_ord);
            F77_CALL(dpotrf)( "L", &dim, G_mat_prop, &dim, &err );
            if( err!= 0 ) error("failure to compute Cholesky factor: metric sampling at proposed point");
            // solve for mean of proposal at theta prop
            for( k=0; k<(*p)*(*neal_ord); k++ ) mu_lvn[k] = grad_prop[k];
            if( p0 < *p ) distlag_collapse_vector( *p, p0, *neal_ord, varind + *q, mu_lvn );
            F77_CALL(dpotrs)( "L", &dim, &nrhs, G_mat_prop, &dim, mu_lvn, &dim, &err ) ;
            if( err!= 0 ) error("failure to solve Cholesky system: inverse on gradient metric sampling at proposed point");
            if( p0 < *p ) distlag_collapse_vector( *p, p0, *neal_ord, varind + *q, transf_prop_theta );
            for( k=0; k<(p0)*(*neal_ord); k++ ){ mu_lvn[k] *= 0.5 * lvn_step * lvn_step ; mu_lvn[k] += transf_prop_theta[k]; }
            if( p0 < *p ) distlag_collapse_vector( *p, p0, *neal_ord, varind + *q, transf_theta );
            lproposal_ratio += distlag_log_pdf_mv_normal( dim, transf_theta, mu_lvn, G_mat_prop, lvn_step );
          }
          
          log_acc_ratio = logpost_prop - logpost_curr + lproposal_ratio ; 
          
          if( log(runif(0.0,1.0)) < log_acc_ratio )
          {
            accepted[0] += 1;
            // swap over all from proposed to current storage
            for( i=0; i<(*neal_ord) * (*p); i++ ) theta[i] = theta_prop[i];
            for( i=0; i<*n; i++ ){ eta[i] = eta_prop[i]; lp2[i] = lp2_prop[i]; }
            for( i=0; i<(*l) * (*m) * (*p); i++ ) wei[i] = wei_prop[i];
            for( i=0; i<(*l) * (*neal_ord) * (*p); i++ ) deriv_wei_curr[i] = deriv_wei_prop[i];
            for( i=0; i<(*n) * (*neal_ord) * (*p); i++ ) deriv_deriv_wx_curr[i] = deriv_deriv_wx_prop[i];
            for( i=0; i<(*n) * (*m) * (*p); i++ ) X_w[i] = X_w_prop[i];
          }
          
        }
        
      }
      
      if( *dl_vars_varsel && *weight_scheme == 1 && iter%10 == 0 )
      {
        proposed[1] += 1;
        // do variable selection for the distributed lag variables in here
        // special case if current model is empty 
        logpost_curr = 0.0;
        logpost_prop = 0.0;
        
        // carry out evals for current model first
        distlag_compute_posterior_quantities_for_joint_sampling_sdl( *n, *q, *p, *m, varind, omega_sq, xi, z, v, W, X_w, prec_pr_sl, prec_pr_sl_sp, post_prec_joint, mu_joint );
        
        // need to do collapse here
        if( q0 < *q || p0 < *p ) 
        {
          distlag_collapse_joint_precision_matrix( *q, *p, q0, p0, *m, varind, post_prec_joint ); 
          distlag_collapse_joint_vector( *q, *p, q0, p0, *m, varind, mu_joint );
        }
        // Cholesky factor in post_prec_beta current
        dim = q0 + p0 * (*m); //Rprintf("\n dim = %d ", dim); 
        F77_CALL(dpotrf)( "L", &dim, post_prec_joint, &dim, &err );
        if( err!= 0 ) error("failure to compute Cholesky factor: variable selection pt. 1"); 
        // compute RHS term
        // solve for mean of full conditional of beta
        F77_CALL(dpotrs)( "L", &dim, &nrhs, post_prec_joint, &dim, mu_joint, &dim, &err );
        if( err!= 0 ) error("failure to solve Cholesky system: variable selection pt. 2");      
        L = post_prec_joint;
        ld = 0.0; b = 0.0;
        for( i=0; i<dim; i++ )
        {
          a = 0.0;
          ld += log( L[i] ) ;
          for( k=i; k<dim; k++ )
          {
            a += L[k] * mu_joint[k] ;
          }
          b += a * a ;
          L += dim ;
        }
        logpost_curr = -ld + 0.5 * b ;
        // add the prior precision terms (assume diagonal)
        C = prec_pr_sl ;
        for( j=0; j<*q; j++ )
        {
          if( varind[j] ) logpost_curr += -0.5 * log( C[0] );
          C += *q + 1;
        }
        C = prec_pr_sl_sp;
        for( j=0; j<*p; j++ )
        {
          for( k=0; k<*m; k++ )
          {
            if( varind[ *q + j ] ) logpost_curr += -0.5 * log( C[0] ) ;
            C += (*p)*(*m) + 1;
          }
        }
        logprior_curr = distlag_compute_logprior( *p, *neal_ord, varind + *q, *weight_scheme, theta, prior_mean_theta, 
                                                  prior_sd_theta, prior_rate_theta) ;
        logpost_curr += logprior_curr; // this is the denominator computed 
        
        // easiest to do first is exp-b-spline
        for( j=0; j<*q + *p; j++ ) varind_prop[j] = varind[j];
        // randomly select variable to swap
        jj = (int) ( runif(0.0,1.0) * (*p) ) ;
        if( varind[ *q + jj ] ) { varind_prop[ *q + jj ] = 0; p0 -= 1; } else { varind_prop[ *q + jj ] = 1; p0 += 1; }
        // PROPOSED STATE
        // now generate the new theta values for the RJ move
        for( i=0; i<(*neal_ord) * (*p); i++ ) theta_prop[i] = theta[i];
        th_ptr = theta_prop + jj * (*neal_ord) ;
        logprop_term = 0.0;
        for( k=0; k<*neal_ord; k++ )
        { 
          if( varind_prop[ *q + jj ] ) 
          {
            th_ptr[k] = rnorm( 0.0, *sd_varsel ); 
            logprop_term -= dnorm( th_ptr[k], 0.0, *sd_varsel, 1 ); 
          }
          else
          {
            logprop_term += dnorm( th_ptr[k], 0.0, *sd_varsel, 1 );
          }
        }
        
        // need to compute the backward gradient
        if( p0 > 0 ) distlag_compute_normed_weight_matrix_and_deriv( *p, *neal_ord, *l, varind_prop + *q, theta_prop, basis_eval, wei_prop, deriv_wei_prop );
       
        // compute the updated linear predictor and reg mats
        if( p0 > 0 )  distlag_compute_weighted_reg_and_dl_linpred( *n, *p, *m, varind_prop + *q, *l, beta, wei_prop, X, X_w_prop, lp2_prop );
        
        distlag_compute_posterior_quantities_for_joint_sampling_sdl( *n, *q, *p, *m, varind_prop, omega_sq, xi, z, v, W, X_w_prop, prec_pr_sl, prec_pr_sl_sp, post_prec_joint, mu_joint );
        
        // need to do collapse here
        if( q0 < *q || p0 < *p ) 
        {
          distlag_collapse_joint_precision_matrix( *q, *p, q0, p0, *m, varind_prop, post_prec_joint ); 
          distlag_collapse_joint_vector( *q, *p, q0, p0, *m, varind_prop, mu_joint );
        }
        // Cholesky factor in post_prec_joint proposed
        dim = q0 + p0 * (*m); 
        F77_CALL(dpotrf)( "L", &dim, post_prec_joint, &dim, &err );
        if( err!= 0 ) error("failure to compute Cholesky factor: variable selection pt. 1"); 
        
        // compute RHS term
        // solve for mean of full conditional 
        F77_CALL(dpotrs)( "L", &dim, &nrhs, post_prec_joint, &dim, mu_joint, &dim, &err );
        if( err!= 0 ) error("failure to solve Cholesky system: variable selection pt. 2");      
        
        L = post_prec_joint;
        ld = 0.0; b = 0.0;
        for( i=0; i<dim; i++ )
        {
          a = 0.0;
          ld += log( L[i] ) ;
          for( k=i; k<dim; k++ )
          {
            a += L[k] * mu_joint[k] ;
          }
          b += a * a ;
          L += dim ;
        }
        logpost_prop = -ld + 0.5 * b ;          
        
        // add the prior precision terms (assume diagonal)
        C = prec_pr_sl ;
        for( j=0; j<*q; j++ )
        {
          if( varind_prop[j] ) logpost_prop += -0.5 * log( C[0] );
          C += *q + 1;
        }
        C = prec_pr_sl_sp;
        for( j=0; j<*p; j++ )
        {
          for( k=0; k<*m; k++ )
          { 
            if( varind_prop[ *q + j ] ) logpost_prop += -0.5 * log( C[0] ) ;
            C += (*p)*(*m) + 1;
          }
        }        
        logprior_prop = distlag_compute_logprior( *p, *neal_ord, varind_prop + *q, *weight_scheme, theta_prop, prior_mean_theta, 
                                                  prior_sd_theta, prior_rate_theta) ;
        logpost_prop += logprior_prop; // this is the denominator computed     
        
        
        log_acc_ratio = logpost_prop - logpost_curr + logprop_term + (2.0 * varind_prop[ *q + jj ] - 1.0) * ( log( varind_prior_probs[ *q + jj ] ) - log( 1.0 - varind_prior_probs[ *q + jj ] ) )  ;
        
        if( log(runif(0.0,1.0)) < log_acc_ratio )
        {
          accepted[1] += 1;
          // swap over all from proposed to current storage
          for( k=0; k<*p + *q; k++ ) varind[k] = varind_prop[k] ;
          
          // if new model accepted then re-sample the beta
          for( i=0; i<*neal_ord; i++ ) theta[ jj * (*neal_ord) + i ] = ( varind[ *q + jj] == 1 ) ? theta_prop[ jj * (*neal_ord) + i ] : 0.0 ;
          for( i=0; i<(*l) * (*m); i++ ) wei[ jj * (*l) * (*m) + i ] = ( varind[ *q + jj] == 1 ) ? wei_prop[jj * (*l) * (*m) + i ] : 0.0 ;
          for( i=0; i<(*l) * (*neal_ord); i++ ) deriv_wei_curr[ jj * (*neal_ord) * (*l) + i ] = deriv_wei_prop[ jj * (*neal_ord) * (*l) + i  ]; // need to check these
          distlag_compute_deriv_WX_eta( *n, *p, *neal_ord, *l, varind+*q, X, deriv_wei_curr, deriv_deriv_wx_curr );
          for( i=0; i<(*n) * (*m) * (*p); i++ ) X_w[i] = X_w_prop[i];
          
          // re-sample beta/gamma in new model- can reuse Cholesky from above
          for( i=0; i<dim; i++ ) njoint[i] = rnorm( 0.0, 1.0 ) ;
          F77_CALL(dtrtrs)( "L", "T", "N", &dim, &nrhs, post_prec_joint, &dim, njoint, &dim, &err );
          if( err!= 0 ) error("failure to solve Cholesky system: posterior joint sampling");
          if( q0 < *q || p0 < *p )
          {
            // expand the vectors
            distlag_expand_joint_vector( *q, *p, q0, p0, *m, varind, mu_joint );
            distlag_expand_joint_vector( *q, *p, q0, p0, *m, varind, njoint );
            dim = *q + *p * (*m) ;
          }
          for( i=0; i<*q; i++ ) gamma[i] = mu_joint[i] + njoint[i] ;
          for( i=0; i<(*p)*(*m); i++ ) beta[i] = mu_joint[*q + i] + njoint[*q + i];
          distlag_update_linear_predictor_1( *n, *q, varind, gamma, W, lp1, eta );
          //distlag_update_linear_predictor_2( *n, *p, *m, varind + *q, beta, X_w, lp2, eta );
          distlag_compute_weighted_reg_and_dl_linpred( *n, *p, *m, varind + *q, *l, beta, wei, X, X_w, lp2 );
          for( i=0; i<*n; i++ ) eta[i] = lp1[i] + lp2[i];
        }
        else
        {
          if( varind[ *q + jj ] ) p0 += 1; else p0 -= 1;
        }
        
      }
      
      // hyperparameter update
      prior_var_ind_prob = rbeta( p0 + hprior_varind[0], *p - p0 + hprior_varind[1] );
      for( k=0; k<*p; k++ ) varind_prior_probs[k] = prior_var_ind_prob ;
      
      // update the adaptive proposals here
      if( iter%(*adapt_interval) == 0 && *weight_scheme < 2 ) 
      {
        b = 1.0/sqrt(iter);
        b = b < 0.01 ? b : 0.01;
        a = (double) accepted[0] / proposed[0] ;
        g0 = ( a < 0.65) ? -1.0 : 1.0 ;
        lvn_step = lvn_step * exp( b * g0 );
      }
      
      if( iter > *burn-1 && (iter - *burn + 1)%(*thin) == 0 )
      {
        // move the storage pointer along
        for( k=0; k<*n; k++ ) eta_s[k] = eta[k];
        eta_s += *n ;
        for( i=0; i<*q; i++ ) gamma_s[i] = gamma[i];
        gamma_s += *q ;
        for( i=0; i<(*m) * (*p); i++ ) beta_s[i] = beta[i];
        beta_s += (*m) * (*p) ;
        if( *weight_scheme < 2 )
        { 
          for( i=0; i<(*p)*(*neal_ord); i++ ) theta_s[i] = theta[i]; 
          theta_s += (*p) * (*neal_ord);
        }
        for( k=0; k< *q + *p; k++ ) var_s[k] = varind[k];
        var_s += (*q + *p);
        if( *num_miss > 0 )
        {
          for( k=0; k<*num_miss; k++ ) miss_samp[k] = miss_holder[k];
          miss_samp += *num_miss;
        }
        loglik_s[0] = distlag_compute_loglik( *n, y, *tau, eta );
        loglik_s += 1;
        prpr_s[0] = varind_prior_probs[0];
        prpr_s += 1;
        if( ( (iter - *burn + 1)/(*thin) ) % 5000 == 0 && *verbose ) Rprintf("\n... completed to sample %d", (iter - *burn + 1)/(*thin) );
      }
      
    }
    
    if( !dl_vars_varsel ) proposed[1] = 1;
    for( kk=0; kk<2; kk++ ) acc_rt_theta[kk] = 100.0 * (double) accepted[kk] / proposed[kk] ; 
    sd_theta_prop[0] = lvn_step;
    
    PutRNGstate();
    
    // no free required when using R_alloc
    return;
}


static const R_CMethodDef cMethods[] = {
  {"distlag_rGIG_half", (DL_FUNC) &distlag_rGIG_half, 4},
  {"distlag_MCMC_bspl", (DL_FUNC) &distlag_MCMC_bspl, 43},
  NULL
};

void R_init_dlvarsel( DllInfo *info )
{
  R_registerRoutines( info, cMethods, NULL, NULL, NULL );
  R_useDynamicSymbols( info, FALSE );
}