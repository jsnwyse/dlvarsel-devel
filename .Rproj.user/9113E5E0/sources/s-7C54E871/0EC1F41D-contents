#ifndef _DISTLAG_UTILS_
#define _DISTLAG_UTILS_

#include "required_libs.h"
#include "distlag_RNG.h"

void distlag_compute_normed_weight_matrix_and_deriv( int p, int ord, int l, int *varind, double *par, double *B, double *W, double *deriv_W );

//void distlag_compute_deriv_normed_weight_matrix( int p, int ord, int l, double *B, double *W, double *deriv_W );

void distlag_compute_weighted_reg_and_dl_linpred( int n, int p, int m, int *varind, int l, double *beta, double *W, double *X, double *XW, double *lp );

void distlag_update_weighted_reg_and_dl_linpred_single_i(int i, int n, int p, int m, int *varind, int l, double *beta, 
                                                         double *W, double *X, double *XW, double *lp );
  
void distlag_compute_posterior_precision_dl( int n, int p, int m, int *varind, double omega_sq, double *v, double *XW, double *prior_prec, double *posterior_prec );

void distlag_compute_posterior_quantities_for_joint_sampling_sdl(int n, int q, int p, int m, int *varind, double omega_sq, double xi, double *z, double *v, double *W, double *Xw, 
                                                            double *prior_prec_stat, double *prior_prec_dl, double *posterior_prec, double *mu_term );

void distlag_update_linear_predictor_1( int n, int q, int *varind, double *gamma, double *W, double *lp1, double *eta );

void distlag_update_linear_predictor_2( int n, int p, int m, int *varind, double *beta, double *Xw, double *lp2, double *eta );

//void distlag_update_linear_predictor_2_single_i(int i, int n, int p, int m, int *varind, double *beta, double *Xw, double *lp2, double *eta );

void distlag_compute_deriv_WX_eta(int n, int p, int ord, int l, int *varind, double *X, double *derivW, double *derivWX_eta );

double distlag_compute_logpost_and_gradient_wrt_theta( int n, int p, int ord, int l, int wei_scheme, int *varind, int *y, double *z, double *v, double omega_sq, double xi, double tau, double *eta,
                                         double *X, double *derivW, double *derivWX_eta, int compute_derivWX_eta, double *beta, double *theta, double *prior_mean_theta, 
                                         double *prior_sd_theta, double *prior_rate_theta, double *grad, int compute_met_tens, double *G_mat  ) ;

/*double distlag_log_post_theta_update( int n, int p, int ord, int wei_scheme, int *y, double tau, double *eta, double *theta, double *prior_mean_theta, 
                                      double *prior_sd_theta, double *prior_rate_theta  ) ; */

void distlag_collapse_precision_matrix( int p, int p0, int m, int *varind, double *prec );

void distlag_collapse_joint_precision_matrix( int q, int p, int q0, int p0, int m, int *varind, double *prec );

void distlag_collapse_vector( int p, int p0, int m, int *varind, double *mu );

void distlag_collapse_joint_vector( int q, int p, int q0, int p0, int m, int *varind, double *mu );
  
void distlag_expand_vector( int p, int p0, int m, int *varind, double *beta );

void distlag_expand_joint_vector( int q, int p, int q0, int p0, int m, int *varind, double *beta );

double distlag_compute_logprior( int p, int ord, int *varind, int wei_scheme, double *theta, double *prior_mean_theta, 
                                 double *prior_sd_theta, double *prior_rate_theta);

double distlag_compute_loglik( int n, int *y, double tau, double *eta );

double distlag_compute_logprior_term( int j, int ord, int wei_scheme, double *theta, double *prior_mean_theta, 
                                      double *prior_sd_theta, double *prior_rate_theta);

void distlag_gradient_wrt_theta_single_variable( int j, int n, int p, int ord, int l, int wei_scheme, int *varind, int *y, double *z, double *v, double omega_sq, double xi, double tau, double *eta,
                                                 double *X, double *derivW, double *derivWX_eta, int compute_derivWX_eta, double *beta, double *theta, double *prior_mean_theta, 
                                                 double *prior_sd_theta, double *prior_rate_theta, double *grad, int compute_met_tens, double *G_mat  );
    
#endif 