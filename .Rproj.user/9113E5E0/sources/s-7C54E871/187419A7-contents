\name{dlvarsel}
\alias{dlvarsel}
\title{Fits a Binary Bayesian Quantile Distributed Lag Model with Variable Selection}
\description{
  \code{dlvarsel} is the main interface used for fitting a binary Bayesian quantile distributed lag model.
}
\usage{
  dlvarsel( 
    y, 
    W = NULL, 
    X = NULL, 
    formula = NULL, 
    tau, 
    l, 
    weight.scheme = c("exp-b-spline","b-spline","nealmon"), 
    m=NULL , 
    knots=NULL,
    prior=NULL,
    start=NULL,
    nsamp = 10^4, 
    burn = 10^3, 
    thin = 1, 
    adapt.interval = 50, 
    sampmeth = c("lang","smmala"), 
    dl.varsel = FALSE,
    verbose = FALSE )
}
\arguments{
  \item{y}{A vector of binary responses corresponding to the response variable. This must be supplied.}
  
  \item{W}{A regression matrix giving non distributed lag variables with each row corresponding to an entry in the response vector \code{y}.}
  
  \item{X}{A matrix giving the distributed lag variables for each individual. More information in the "Details" section. }
  
  \item{formula}{Functionality for formulae, not currently fully functional and use is not currently recommended.}
  
  \item{tau}{The quantile of the response distribution that the model will be fitted to. The default is 0.5, corresponding to median regression.}
  
  \item{l}{The length of the time windows.}
  
  \item{weight.scheme}{Either \code{exp-b-spline} (the default), \code{b-spline} or \code{nealmon}. This tells the algorithm how the distributed lag functions are constructed. }
  
  \item{m}{The degree of the B-spline basis in the weighting scheme. Defaults to cubic as in \code{splines} package.}
  
  \item{knots}{The positions within interval \code{(0,l)} to place knots for the B-spline basis. If unspecified this defaults to nine evenly spaced knots i.e. splitting \code{(0,l)} into ten intervals of equal length.}
  
  \item{prior}{A named list that must contain elements named \code{precision_dynamic} and \code{precision_static}. Both must be square matrices (or a scalar, where appropriate). See "Details" for more information.}
  
  \item{start}{A named list with one element named \code{init.theta}, a vector whose length is equal to the number of columns of \code{dynamic} multipled by \code{ord_nealmon}. This indicates the inital value of the theta parameters in the MCMC. More information in the "Details" section.}
  
  \item{nsamp}{The total number of total MCMC iterations, including burn-in.}
  
  \item{burn}{The number of MCMC iterations discarded as burn in.}
  
  \item{thin}{The multiple of which MCMC iterations post burn-in are retained, for the purpose of reducing autocorrelation in the final posterior sample.}
  
  \item{adapt.interval}{The rate at which stepsizes for \code{sampmeth} are adapted.}
  
  \item{sampmeth}{The sampling method for the distributed lag parameters in the case of \code{nealmon} or \code{exp-b-spline} weightings. This can be either \code{lang} or \code{smmala} corresponding to Langevin or Simplified Manifold Metrolopolis Adjusted Langevin sampling. Adaptive stepsizes are used.  }
  
  \item{dl.varsel}{Logical determining whether variable selection should be carried out. If \code{TRUE}, the variables included become part of the Bayesian model.}
  
  \item{verbose}{Logical indicating whether a progress update on sampling should be printed during the run.}
  
}
\value{
  Returns a list containing the following elements:
    
    \item{samples}{Itself a list containing the posterior sample of each of the parameters.}
  \item{langstep}{}
  \item{acc.rt}{The acceptance rate of the theta proposals in the MCMC algorithm and the variable selection moves.}
  
}
\details{
  
  If there are \code{n} responses, then \code{X} will be a matrix with \code{n} columns and \code{p * l} rows i.e. a column contains all distributed lag variables for a single individual. Care should be taken with the ordering of entries in columns: entries should be ordered chronologically most recent to past by row. 
  
  The valid options for the \code{weight_scheme} argument are "nealmon", "b-spline" and "exp-b-spline". "nealmon" results in a parametric Exponential Almon DLF of order given by the nealmon.ord argument. "b-spline" is a non-parametric DLF using b-splines. "exp-b-spline" also uses b-splines, but the basis polynomials are normalised so that the resulting weights sum to one.
  
  The \code{prior} argument should be a list with two named elements: \code{precision_static} and \code{precision_dynamic}, the precision matrix of (respectively) the static and dynamic parameter priors. \code{precision_static} should be a square matrix whose dimension is equal to the number of static variables, and likelwise \code{precision_dynamic} is a square matrix with dimensions equal to the number of dynamic variables times the number of basis functions \code{m} if basis functions are used. \code{NULL} is also an accepted input, and used as the default value, in which case a diagonal matrix is used with diagonal elements equal to 1 except for the intercept which has diagonal element of 0.25.
  
  The \code{start} argument currently takes a list as input with only one named element, \code{init.theta}, which corresponds to the starting value for the theta parameters. Users are currently unable to supply starting values for other parameters, but we wish to add this in the future.
  \code{init.theta} should be a vector of length the number of variables in \code{X} times \code{m} in the case of "exp-b-spline" weighting and times 2 for "nealmon" weighting. 
}
\author{Daniel Dempsey and Jason Wyse}
