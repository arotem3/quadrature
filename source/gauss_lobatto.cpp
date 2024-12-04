#include "quadrature.hpp"

extern "C" void dsteqr_(char *COMPZ, int *N, double *D, double *E, double *Z, int *LDZ_dummy, double *WORK, int *INFO);

void gauss_lobatto_impl(int n, double *x)
{
  std::fill_n(x, n, 0.0);

  std::vector<double> E(n - 3);
  for (int i = 0; i < n - 3; ++i)
  {
    double I = i + 1;
    E.at(i) = std::sqrt(I * (I + 2.0) / (2.0 * I + 3.0) / (2.0 * I + 1.0));
  }

  int N = n - 2;
  int info;
  char compz = 'N';
  int LDZ_dummy = 1;

  dsteqr_(&compz, &N, x + 1, E.data(), nullptr, &LDZ_dummy, nullptr, &info);

  if (info != 0)
    throw std::runtime_error("gauss_lobatto failed to compute eigenvalues of companion matrix.");

  x[0] = -1.0;
  x[n - 1] = 1.0;

  // refine roots with Newton's method
  for (int i = 1; i < n / 2; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      double P = quad::jacobiP(n - 2, 1, 1, x[i]);
      double dP = quad::jacobiP_derivative(1, n - 2, 1, 1, x[i]);
      x[i] -= P / dP;
    }

    x[n - 1 - i] = -x[i];
  }

  if (n & 1)
    x[n / 2] = 0.0;
}

namespace quad
{
  void gauss_lobatto(int n, double *x, double *w)
  {
    if (n < 2)
      throw std::invalid_argument("gauss_lobatto only computes quadrature rules with 2 or more points.");

    constexpr double nodes[][9] = {
        {-1, 1},
        {-1, 0, 1},
        {-1, -0.447213595499958, 0.447213595499958, 1},
        {-1, -0.654653670707977, 0, 0.654653670707977, 1},
        {-1, -0.765055323929465, -0.285231516480645, 0.285231516480645, 0.765055323929465, 1},
        {-1, -0.830223896278567, -0.468848793470714, 0.0, 0.468848793470714, 0.830223896278567, 1},
        {-1, -0.871740148509607, -0.591700181433142, -0.209299217902479, 0.2092992179024789, 0.591700181433142, 0.871740148509607, 1},
        {-1, -0.899757995411460, -0.677186279510738, -0.363117463826178, 0, 0.363117463826178, 0.677186279510738, 0.899757995411460, 1}};

    if (n <= 9)
      std::copy_n(nodes[n - 2], n, x);
    else
      gauss_lobatto_impl(n, x);

    for (int i = 0; i < n; ++i)
    {
      double P = jacobiP(n - 1, 0, 0, x[i]);
      w[i] = 2.0 / (n * (n - 1) * P * P);
    }
  }
} // namespace quad
