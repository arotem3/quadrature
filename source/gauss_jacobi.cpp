#include "quadrature.hpp"
#include <iostream>

extern "C" int dsteqr_(char *COMPZ, int *N, double *D, double *E, double *Z, int *LDZ_dummy, double *WORK, int *INFO);

namespace quad
{
  void gauss_jacobi(int n, double a, double b, double *x, double *w)
  {
    if (n < 1)
      throw std::invalid_argument("gauss_jacobi only computes quadrature rules with 1 or more points.");

    if (a <= -1 || b <= -1)
      throw std::invalid_argument("The Jacobi polynomials are not defined for a <= -1 or b <= -1.");

    x[0] = (b - a) / (2 + a + b);
    if (a + b == 0)
    {
      std::fill_n(x + 1, n - 1, 0.0);
    }
    else
    {
      for (int i = 1; i < n; ++i)
      {
        double p = b * b - a * a;
        double q = (2 * i + a + b) * (2 * i + 2 + a + b);
        x[i] = p / q;
      }
    }

    std::vector<double> E(n - 1);
    E[0] = 2.0 / (2.0 + a + b) * std::sqrt((1.0 + a) * (1.0 + b) / (3.0 + a + b));
    for (int i = 1; i < n - 1; ++i)
    {
      double k = i + 1;
      E[i] = 2.0 / (2.0 * k + a + b) * std::sqrt((k + a) * (k + b) / (2 * k + a + b + 1)) * std::sqrt(k * (k + a + b) / (2.0 * k + a + b - 1));
    }

    int info;
    char compz = 'N';
    int LDZ_dummy = 1;

    dsteqr_(&compz, &n, x, E.data(), nullptr, &LDZ_dummy, nullptr, &info);

    if (info != 0)
      throw std::runtime_error("gauss_jacobi failed to compute eigenvalues of companion matrix.");

    for (int i = 0; i < n; ++i)
    {
      for (int j = 0; j < 3; ++j)
      {
        double P = jacobiP(n, a, b, x[i]);
        double dP = jacobiP_derivative(1, n, a, b, x[i]);
        x[i] -= P / dP;
      }

      double P = jacobiP(n - 1, a, b, x[i]);
      double dP = jacobiP_derivative(1, n, a, b, x[i]);
      w[i] = 1.0 / (P * dP);
    }

    double s = 0.0;
    for (int i = 0; i < n; ++i)
      s += w[i];

    double m = std::log(2.0) * (a + b + 1.0) + std::lgamma(a + 1.0) + std::lgamma(b + 1.0) - std::lgamma(2.0 + a + b);
    m = std::exp(m) / s;

    for (int i = 0; i < n; ++i)
      w[i] *= m;
  }
} // namespace quad
