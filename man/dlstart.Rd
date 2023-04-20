\name{dlstart}
\alias{dlstart}
\title{Generate Initial Values for Distributed Lag MCMC}
\description{
  \code{dlstart} returns a set of initial values for the MCMC run.
}
\usage{
dlstart( 
p, 
m, 
prior, 
weight.scheme )
}
\arguments{
  \item{p}{The number of distributed lag variables determined from \code{X} using \code{l}}
  
  \item{m}{The number of basis functions used when using b-splines in the weighting scheme or 2 if using \code{nealmon} weighting. Otherwise forced to 1.}
  
  \item{prior}{A list returned from \code{dlprior} or one of an identical nature to allow initialization using the prior.}
  
  \item{weight.scheme}{Either \code{exp-b-spline} (the default), \code{b-spline} or \code{nealmon}. This tells the algorithm how the distributed lag functions are constructed. }
  
}
\value{
  Returns a list with one element containing initial values for the chain.
}

\author{Daniel Dempsey and Jason Wyse}
