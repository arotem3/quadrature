#ifndef __QUADRATURE_HPP__
#define __QUADRATURE_HPP__

#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <vector>
#include <unordered_map>

namespace quad
{
  /**
   * @brief Computes the n-point Gauss-Legendre quadrature rule. This rule is
   * exact for polynomials up to degree 2*n-1.
   *
   * @param n number of points.
   * @param x On exit, the nodes of the quadrature rule.
   * @param w On exit, the weights of the quadrature rule.
   */
  void gauss_legendre(int n, double *x, double *w);

  /**
   * @brief Computes the n-point Gauss-Lobatto quadrature rule. This rule is
   * exact for polynomials up to degree 2*n-3.
   *
   * @param n number of points.
   * @param x On exit, the nodes of the quadrature rule.
   * @param w On exit, the weights of the quadrature rule.
   */
  void gauss_lobatto(int n, double *x, double *w);

  /**
   * @brief Computes the n-point Gauss-Jacobi quadrature rule. This rule is
   * exact for polynomials up to degree 2*n-1 with weight function (1-x)^a * (1+x)^b.
   *
   * @param n number of points.
   * @param x On exit, the nodes of the quadrature rule.
   * @param w On exit, the weights of the quadrature rule.
   */
  void gauss_jacobi(int n, double a, double b, double *x, double *w);

  /**
   * @brief evaluates the Jacobi polynomial P{m; a, b}(x) given P{m-1; a, b}(x) and P{m-2; a, b}(x).
   *
   * The Jacobi polynomial P{m; a, b} is of degree m. They are orthgonal with weight (1-x)^a * (1+x)^b.
   *
   * @param m degree of the polynomial.
   * @param a
   * @param b
   * @param x evaluation point.
   * @param y1 P{m-1; a, b}(x).
   * @param y2 P{m-2; a, b}(x).
   * @return P{m; a, b}(x).
   */
  static inline constexpr double jacobiP_next(unsigned int m, double a, double b, double x, double y1, double y2)
  {
    constexpr double one = 1.0, two = 2.0;
    double yp1 = (two * m + a + b - one) * ((two * m + a + b) * (two * m + a + b - two) * x + a * a - b * b) * y1 - two * (m + a - one) * (m + b - one) * (two * m + a + b) * y2;
    yp1 /= two * m * (m + a + b) * (two * m + a + b - two);
    return yp1;
  }

  /**
   * @brief evaluates the Jacobi polynomial P{n; a, b}(x).
   *
   * The Jacobi polynomial P{m; a, b} is of degree m. They are orthgonal with weight (1-x)^a * (1+x)^b.
   *
   * @param n degree of the polynomial.
   * @param a
   * @param b
   * @param x evaluation point.
   * @return P{m; a, b}(x).
   */
  static inline constexpr double jacobiP(unsigned int n, double a, double b, double x)
  {
    constexpr double one = 1.0, half = 0.5, two = 2.0;
    double ym1 = 1;

    if (n == 0)
      return ym1;

    double y = (a + one) + half * (a + b + two) * (x - one);

    for (unsigned int m = 2; m <= n; ++m)
    {
      double yp1 = jacobiP_next(m, a, b, x, y, ym1);
      ym1 = y;
      y = yp1;
    }

    return y;
  }

  /**
   * @brief evaluates the k-th derivative of the Jacobi polynomial P{n; a, b}(x).
   *
   * @param k the order of the derivative.
   * @param n the degree of the polynomial.
   * @param a
   * @param b
   * @param x evaluation point.
   * @return (d/dx)^k P{n; a, b}(x).
   */
  static inline constexpr double jacobiP_derivative(unsigned int k, unsigned int n, double a, double b, double x)
  {
    if (k > n)
      return 0.0;
    else
    {
      constexpr double one = 1.0, two = 2.0;
      double s = std::lgamma(n + a + b + one + k) - std::lgamma(n + a + b + one) - k * std::log(two);
      return std::exp(s) * jacobiP(n - k, a + k, b + k, x);
    }
  }

} // namespace quad

#endif