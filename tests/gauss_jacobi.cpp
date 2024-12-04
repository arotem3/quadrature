#include "quadrature.hpp"
#include <iostream>
#include <iomanip>

// This test uses the fact that the Jacobi polynomial P{n; -1/2, -1/2} is
// proportial to the Chebyshev polynomial T{n}. The roots of the Chebyshev nodes
// are known, so we compare them to those computed by gauss_jacobi().
static int test_gauss_cheb(int n)
{
  constexpr double tol = 1e-12;
  std::vector<double> x(n), w(n);

  quad::gauss_jacobi(n, -0.5, -0.5, x.data(), w.data());

  double error = 0.0;
  for (int i = 0; i < n; ++i)
  {
    double t = -std::cos((2 * i + 1) * M_PI / (2 * n));
    double e = std::abs(t - x[i]);

    e += std::abs(M_PI / n - w[i]);
    error = std::max(error, e);
  }

  if (error > tol)
  {
    std::cout << "test_gauss_cheb(" << n << ") failed with error " << error << "\n";
    return 1;
  }

  return 0;
}

// Compare against gauss_legendre(n) which corresponds to a = b = 0.
static int test_gauss_jacobi_legendre(int n)
{
  constexpr double tol = 1e-12;

  std::vector<double> x(n), w(n);
  quad::gauss_jacobi(n, 0.0, 0.0, x.data(), w.data());

  std::vector<double> t(n), s(n);
  quad::gauss_legendre(n, t.data(), s.data());

  double error = 0.0;
  for (int i = 0; i < n; ++i)
  {
    double e = std::abs(x[i] - t[i]) + std::abs(w[i] - s[i]);
    error = std::max(error, e);
  }

  if (error > tol)
  {
    std::cout << "test_gauss_jacobi_legendre(" << n << ") failed with error " << error << "\n";
    return 1;
  }

  return 0;
}

// Check correctness when a + b = 0, e.g. a = -1/2, b = 1/2 for which we have an
// exact answer for the integral.
static int test_gauss_jacobi_abzero(int n)
{
  constexpr double tol = 1e-12;
  std::vector<double> x(n), w(n);

  quad::gauss_jacobi(n, -0.5, 0.5, x.data(), w.data());

  double I = 0.0;
  std::cout << std::setprecision(16);
  for (int i = 0; i < n; ++i)
  {
    double p = std::pow(x[i], 2 * n - 1);
    I += p * w[i];
  }

  double exact = std::sqrt(M_PI) * std::exp(std::lgamma(0.5 + n) - std::lgamma(1.0 + n));
  double error = std::abs(I - exact) / exact;

  if (error > tol)
  {
    std::cout << "test_gauss_jacobi_abzero(" << n << ") failed with error " << error << "\n";
    return 1;
  }

  return 0;
}

// check for somewhat arbitrary values a = 3, b = 2 for which we can easily
// compute an exact answer for the integral.
static int test_gauss_jacobi(int n)
{
  constexpr double tol = 1e-12;
  std::vector<double> x(n), w(n);

  quad::gauss_jacobi(n, 2.0, 3.0, x.data(), w.data());

  double I = 0.0;
  std::cout << std::setprecision(16);
  for (int i = 0; i < n; ++i)
  {
    double p = std::pow(x[i], 2 * n - 1);
    I += p * w[i];
  }

  double exact = 16.0 / (2.0 * n + 1.0) / (2.0 * n + 3.0) / (2.0 * n + 5.0);
  double error = std::abs(I - exact) / exact;

  if (error > tol)
  {
    std::cout << "test_gauss_jacobi(" << n << ") failed with error " << error << "\n";
    return 1;
  }

  return 0;
}

int main()
{
  int fails = 0;

  for (int n = 2; n < 10; ++n)
    fails += test_gauss_cheb(n);
  fails += test_gauss_cheb(50);

  for (int n = 2; n < 10; ++n)
    fails += test_gauss_jacobi_legendre(n);
  fails += test_gauss_jacobi_legendre(50);

  for (int n = 2; n < 10; ++n)
    fails += test_gauss_jacobi_abzero(n);
  fails += test_gauss_jacobi_abzero(50);

  for (int n = 2; n < 10; ++n)
    fails += test_gauss_jacobi(n);
  fails += test_gauss_jacobi(50);

  return fails;
}