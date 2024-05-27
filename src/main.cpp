#include <iostream>
#include <ranges>

#include "model.h"
#include "monte_carlo.h"
#include "pde.h"

constexpr double expiry_time = 100.0 / 365.0;
constexpr int monte_carlo_simulation_count = 20000;
constexpr int monte_carlo_time_step_count = 20000;
constexpr int pde_time_step_count = 60;

// Seed for the pseudo-random number generator. Set here for reproducibility.
constexpr unsigned int seed = 20240524;

ublas::c_matrix<double, 3, 3> example_volatility_matrix(const double t) {
  ublas::c_matrix<double, 3, 3> coefficients{};
  for (auto i : std::views::iota(0, 3)) {
    for (auto j : std::views::iota(0, 3)) {
      // We introduce some shifts so entries change differently over time.
      int shift1 = i + j % 5 - 2;
      int shift2 = i + j % 4 - 2;
      coefficients(i,j) = 0.05
        + 0.01 * std::pow(t - 5 + shift1, 2)
        + 0.1 * shift2;
    }
  }
  return coefficients;
}

double example_riskfree_rate(const double t) {
  return 0.025 + t * 0.01;
}

template <int D>
void test(const Model<D>& model, const BasketOption<D>& basket_option) {
  double initial_basket_value =
    ublas::inner_prod(basket_option.weights, model.initial_prices);
  std::cout << "Initial basket value: " << initial_basket_value << std::endl;
  std::cout << "Option:               " << basket_option << std::endl;

  double monte_carlo_price = MonteCarlo::pricer(
    seed,  monte_carlo_simulation_count, monte_carlo_time_step_count,
    model, basket_option);
  std::cout << "Monte Carlo price:    " << monte_carlo_price << std::endl;

  double pde_price = PDE::pricer(pde_time_step_count, model, basket_option);
  std::cout << "PDE price:            " << pde_price << std::endl;
}

int main(int argc, char *argv[]) {
  ublas::c_vector<double, 3> initial_prices{};
  initial_prices(0) = 100.0;
  initial_prices(1) = 20.0;
  initial_prices(2) = 50.0;
  const Model<3> model{
    initial_prices,
    example_volatility_matrix,
    example_riskfree_rate
  };

  ublas::c_vector<double, 3> weights1{};
  weights1(0) = 2.5;
  weights1(1) = 3.0;
  weights1(2) = 1.0;
  const BasketOption<3> basket_option1{weights1, expiry_time, 250.0};

  ublas::c_vector<double, 3> weights2{};
  weights2(0) = 4.0;
  weights2(1) = 1.5;
  weights2(2) = 6.0;
  const BasketOption<3> basket_option2{weights2, expiry_time, 850.0};

  std::cout << "=== Test for basket option in the money ===\n" << std::endl;
  test(model, basket_option1);
  std::cout << "\n=== Test for basket option out of the money ===\n" << std::endl;
  test(model, basket_option2);

  return 0;
}
