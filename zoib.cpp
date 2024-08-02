#include <TMB.hpp>

/* zero-one-inflated beta log-PDF of a single response
 * Args:
 *   y: response value
 *   mu: mean parameter of the beta part
 *   phi: precision parameter of the beta part
 *   zoi: zero-one-inflation probability
 *   coi: conditional one-inflation probability
 * Returns:
 *   a scalar to be added to the log posterior
 */

template <class T>
T zero_one_inflated_beta_lpdf(T y, T mu, T phi, T zoi, T coi)
{
  vector<T> shape(2);
  T out;
  shape << mu * phi, (1 - mu) * phi;
  if (y == 0 || y == 1)
    out = dbinom(T(1), T(1), zoi, true) + dbinom(y, T(1), coi, true);
  else
    out = dbinom(T(0), T(1), zoi, true) + dbeta(y, shape[0], shape[1], true);
  return out;
}

template <class Type>
Type objective_function<Type>::operator()()
{
  parallel_accumulator<Type> f(this);

  DATA_SCALAR(prior_only);
  DATA_VECTOR(y);
  DATA_MATRIX(X);
  DATA_SPARSE_MATRIX(IID);

  PARAMETER_VECTOR(betas);
  f -= dnorm(betas, Type(0), Type(1), true).sum();

  PARAMETER(log_phi);
  PARAMETER(logit_zoi);
  PARAMETER(logit_coi);

  Type
      phi = exp(log_phi),
      zoi = invlogit(logit_zoi),
      coi = invlogit(logit_coi);

  f -= dnorm(log_phi, Type(0), Type(1), true) + log_phi;

  f -= log(zoi) + log(1 - zoi); // change of variables: logit
  f -= dbeta(zoi, Type(1), Type(20), true);

  f -= log(coi) + log(1 - coi);
  f -= dbeta(coi, Type(0.5), Type(0.5), true);

  PARAMETER_VECTOR(pid);
  PARAMETER(log_sd_pid);
  f -= dnorm(log_sd_pid, Type(0), Type(1), true) + log_sd_pid;

  f -= dnorm(pid, Type(0), exp(log_sd_pid), true).sum();
  f -= dnorm(pid.sum(), Type(0), Type(0.001), true);

  int N = y.size();

  vector<Type> mu = X * betas + IID * pid;
  mu = invlogit(mu);

  vector<Type> lli(N);
  for (int i = 0; i < N; i++)
    f -= zero_one_inflated_beta_lpdf<Type>(y[i], mu[i], phi, zoi, coi);
  REPORT(mu);
  REPORT(coi);
  REPORT(zoi);
  REPORT(phi);
  REPORT(betas);
  REPORT(pid);

  return f;
}
