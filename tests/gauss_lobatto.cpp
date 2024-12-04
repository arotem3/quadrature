#include "quadrature.hpp"
#include <iostream>

// Chebyshev polynomial.
static double T(int n, double x)
{
  return std::cos(n * std::acos(x));
}

// Sequence of degree polynomials which always integrates to 2 over [-1,1].
static double f(int n, double x)
{
  double a = 1 - n * n;
  double b = 1 - (n - 1) * (n - 1);
  return a * T(n, x) + b * T(n - 1, x);
}

static int test_gauss_lobatto(int n)
{
  constexpr double tol = 1e-10;

  std::vector<double> x(n), w(n);
  quad::gauss_lobatto(n, x.data(), w.data());

  const int p = 2 * n - 3;

  double I = 0.0;
  for (int i = 0; i < n; ++i)
    I += w[i] * f(p, x[i]);

  const double error = std::abs(I - 2.0);

  if (error < tol)
    return 0;
  else
  {
    std::cout << "test_gauss_lobatto(" << n << ") failed with error " << error << "\n";
    return 1;
  }
}

int main()
{
  int fails = 0;

  for (int n = 2; n < 15; ++n)
  {
    fails += test_gauss_lobatto(n);
  }
  fails += test_gauss_lobatto(50);

  return fails;
}