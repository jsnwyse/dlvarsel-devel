dlcdfALD <- function( x, tau, mu, sigma )
{
  z <- (x - mu)/sigma
  pr <- tau * exp( (1-tau) * pmin(z,0) ) + (1-tau) * (1 - exp(-tau  * pmax(z,0)))
  return(pr)
}