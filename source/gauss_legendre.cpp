#include "quadrature.hpp"

extern "C" void dsteqr_(char *COMPZ, int *N, double *D, double *E, double *Z, int *LDZ_dummy, double *WORK, int *INFO);

void gauss_legendre_impl(int n, double *x)
{
  std::fill_n(x, n, 0.0);

  std::vector<double> E(n - 1);
  for (int i = 0; i < n - 1; ++i)
  {
    double k = i + 1;
    E.at(i) = k * std::sqrt(1.0 / (4.0 * k * k - 1.0));
  }

  int info;
  char compz = 'N';
  int LDZ_dummy = 1;

  dsteqr_(&compz, &n, x, E.data(), nullptr, &LDZ_dummy, nullptr, &info);

  if (info != 0)
    throw std::runtime_error("gauss_legendre failed to compute eigenvalues of companion matrix.");

  for (int i = 0; i < n / 2; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      double P = quad::jacobiP(n, 0, 0, x[i]);
      double dP = quad::jacobiP_derivative(1, n, 0, 0, x[i]);
      x[i] -= P / dP;
    }

    x[n - 1 - i] = -x[i];
  }

  if (n & 1)
    x[n / 2] = 0.0;
}

namespace quad
{
  void gauss_legendre(int n, double *x, double *w)
  {
    if (n < 1)
      throw std::invalid_argument("gauss_legendre only computes quadrature rules with 1 or more points.");

    double nodes[][10] = {
        {0.0},
        {-0.577350269189625764509149, 0.577350269189625764509149},
        {-0.774596669241483377035853, 0.0, 0.774596669241483377035853},
        {-0.861136311594052575223946, -0.339981043584856264802666, 0.339981043584856264802666, 0.861136311594052575223946},
        {-0.906179845938663992797627, -0.538469310105683091036314, 0.0, 0.538469310105683091036314, 0.906179845938663992797627},
        {-0.932469514203152027812302, -0.661209386466264513661400, -0.238619186083196908630502, 0.238619186083196908630502, 0.661209386466264513661400, 0.932469514203152027812302},
        {-0.949107912342758524526190, -0.741531185599394439863865, -0.405845151377397166906606, 0.0, 0.405845151377397166906606, 0.741531185599394439863865, 0.949107912342758524526190},
        {-0.960289856497536231683561, -0.796666477413626739591554, -0.525532409916328985817739, -0.183434642495649804939476, 0.183434642495649804939476, 0.525532409916328985817739, 0.796666477413626739591554, 0.960289856497536231683561},
        {-0.968160239507626089835576, -0.836031107326635794299430, -0.613371432700590397308702, -0.324253423403808929038538, 0.0, 0.324253423403808929038538, 0.613371432700590397308702, 0.836031107326635794299430, 0.968160239507626089835576},
        {-0.973906528517171720077964, -0.865063366688984510732097, -0.679409568299024406234327, -0.433395394129247190799266, -0.148874338981631210884826, 0.148874338981631210884826, 0.433395394129247190799266, 0.679409568299024406234327, 0.865063366688984510732097, 0.973906528517171720077964}};

    if (n <= 10)
      std::copy_n(nodes[n - 1], n, x);
    else
      gauss_legendre_impl(n, x);

    for (int i = 0; i < n; ++i)
    {
      double P = jacobiP_derivative(1, n, 0, 0, x[i]);
      w[i] = 2.0 / (1.0 - x[i] * x[i]) / (P * P);
    }
  }
} // namespace quad
