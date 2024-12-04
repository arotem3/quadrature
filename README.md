# quadrature
Algorithms for computing Gauss quadrature rules.

There are three methods implemented:
* `gauss_legendre`: For approximating integrals of the form: $\int_{-1}^1 f(x) dx$. Gauss-Legendre rules do not include the end-points $\pm 1$. A Gauss-Legendre rule with $n$ points is exact for any $f$ which is a polynomial of degree $2n-1$ or less.
* `gauss_lobatto`: For approximating integrals of the form: $\int_{-1}^1 f(x) dx.$ Gauss-Lobatto rules include the end-points $\pm 1$. A Gauss-Lobatto rule with $n$ points is exact for any $f$ which is a polynomial of degree $2n-3$ or less.
* `gauss_jacobi`: For approximating integrals of the form $\int_{-1}^1 (1-x)^a (1+x)^b f(x) dx,$ where $a > -1, b > -1$. Gauss-Jacobi rules do not include the end-points $\pm 1$. A Gauss-Jacobi rule with $n$ points is exact for any $f$ which is a polynomial of degree $2n-1$ or less.

## Example
```c++
#include <iostream>
#include "quadrature.hpp"

using namespace quad;

double f(double x);

int main()
{
  const int n = 10;
  double x[n], w[n];

  gauss_legendre(n, x, w);

  double integral = 0.0;
  for (int i = 0; i < n; ++i)
    integral += w[i] * f(x[i]);

  std::cout << "integral ~ " << integral << std::endl;
};
```
