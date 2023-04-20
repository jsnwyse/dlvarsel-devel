###### Default Prior for Distributed Lag parameters

dlprior <- function( p, q, m, l, weight.scheme, basis.eval ) {
  
  prior <- list()
  
  if( weight.scheme == "b-spline" )
  {
    prior$prior.mean.theta <- rep(NULL,m)
    prior$prior.sd.theta <- rep(NULL,m)
    prior$prior.rate.theta <- rep(NULL,m)
    prior$precision_static <- diag( 1/c(100, rep( 100, q-1 )) )
    prior$precision_dynamic <- diag( 1/100, m * p )
    
    # experimetnal prior on dl basis
    eta.1 <- -0.1
    eta.2 <- -0.01
    sigsq <- 1^2
    
    V <- diag( exp(eta.1*(0:(l-1))), nrow=l )
    M <- diag( exp(eta.2*(0:(l-1))), nrow=l )
    Id <- diag( rep(1,l), nrow=l )
    J <- matrix( 1, nrow=l, ncol=l )
    
    W.cov <- M %*% t(M)  + (Id - M) %*% ( J %*% t(Id - M) )
    wvals <- diag( W.cov )
    sqrt.prec <- 1/sqrt(wvals)
    W.corr <- W.cov * ( sqrt.prec %*% t(sqrt.prec) )
    
    Omega <- sigsq * ( V %*% ( W.corr %*% V ) )
    QR.B <- qr(basis.eval)
    GammaUt <- qr.solve( QR.B, Omega )
    Gammat <- qr.solve( QR.B, t(GammaUt) )
    Gamma <- t(Gammat)
    Gamma <- chol(Gamma)
    prior$precision_dynamic <- chol2inv(Gamma)
  }
  if( weight.scheme == "exp-b-spline" )
  {
    prior$prior.mean.theta <- rep( rep(0,m), p ) 
    prior$prior.sd.theta <- rep( rep(1,m), p )
    prior$prior.rate.theta <- rep(NULL,2*p)
    sigsq <- 1
    prior$precision_static <- diag( 1/c(4, rep( sigsq, q-1 )) )
    prior$precision_dynamic <- diag( 1/sigsq, m * p )
  }
  if( weight.scheme == "nealmon" ) 
  {
    prior$prior.mean.theta <- rep( l/10, 2*p ) 
    prior$prior.sd.theta <- rep( (l/10)/4 , 2*p )
    prior$prior.rate.theta <- rep(10,2*p)
    sigsq <- 1
    prior$precision_static <- diag( 1/c(4, rep( sigsq, q-1 )) )
    prior$precision_dynamic <- diag( 1/sigsq, m * p )
  }
  
  return(prior)
}

dlstart <- function( p, m, prior, weight.scheme ) {
  
  if ( p == 0 ) { return(NULL) }

  if( weight.scheme == "b-spline" ) return(NULL)
  
  if( weight.scheme == "exp-b-spline" ) init.theta <- rnorm( length(prior$prior.mean.theta), 0, 1 )

  if( weight.scheme == "nealmon" ) 
  {
    init.theta <- rnorm( 2*p, prior$prior.mean.theta, prior$prior.sd.theta )
    idx <- seq(2, 2*p, by=2)
    init.theta[ idx ] <- - rexp( length(idx), rate = prior$prior.rate.theta[idx] )
  }
  
  return( list(init.theta=init.theta) )  
}

