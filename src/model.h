#pragma once

#include <functional>
#include <ranges>

#include <boost/numeric/ublas/matrix.hpp>

namespace ublas = boost::numeric::ublas;

// Matrix of volatility coefficients as a function of time. (C in PDF)
template <int D>
using volatility_matrix = std::function<ublas::c_matrix<double, D, D>(const double)>;

// Risk-free rate as a function of time. (r in PDF)
using interest_rate = std::function<double(const double)>;

template <int D>
struct Model {
  ublas::c_vector<double, D> initial_prices; // S(0)
  volatility_matrix<D> volatility; // C(t)
  interest_rate riskfree_rate; // r(t)
};

template <int D>
struct BasketOption {
  ublas::c_vector<double, D> weights; // w
  double expiry_time; // T
  double strike_price; // K
};
