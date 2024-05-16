#include <iostream>

#include "model.h"
#include "monte_carlo.h"
#include "pde.h"

constexpr int expiry_time = 10;
constexpr int asset_count = 3;
constexpr int monte_carle_path_count = 10000;
constexpr int pde_step_count = 100;

ublas::c_matrix<double, asset_count, asset_count> example_volatility_matrix(const double t) {
  ublas::c_matrix<double, asset_count, asset_count> coefficients{};
  for (auto i : std::views::iota(0, asset_count)) {
    for (auto j : std::views::iota(0, asset_count)) {
      // We introduce some shifts so entries change differently over time.
      int shift1 = i + j % 5 - 2;
      int shift2 = i + j % 4 - 2;
      coefficients(i,j) = 0.01 * std::pow(t - 5 + shift1, 2) + 0.1 * shift2;
    }
  }
  return coefficients;
}

double example_riskfree_rate(const double t) {
  return 0.01 + t * 0.002;
}

int main(int argc, char *argv[]) {
  // Example instance of the local volatility model with time-dependent only volatily coefficients.
  ublas::c_vector<double, 3> initial_prices{};
  initial_prices(0) = 100;
  initial_prices(1) = 20;
  initial_prices(2) = 50;
  const Model<3> model{initial_prices, example_volatility_matrix, example_riskfree_rate};

  // Example 1
  {
    ublas::c_vector<double, 3> weights{};
    weights(0) = 0.2;
    weights(1) = 0.3;
    weights(2) = 1 - weights(0) - weights(1);
    const BasketOption<3> basket_option{weights, expiry_time, 65};

    double monte_carlo_price = monte_carlo_pricer(model, basket_option);
    double pde_price = pde_pricer(model, basket_option);

    std::cout << "Basket option 1: " << basket_option << "\n" <<
        "\tMonte Carlo price: " << monte_carlo_price << "\n" <<
        "\tPDE price:         " << pde_price << "\n" << std::endl;
  }

  // Example 2
  {
    ublas::c_vector<double, 3> weights{};
    weights(0) = 0.4;
    weights(1) = 0.2;
    weights(2) = 1 - weights(0) - weights(1);
    const BasketOption<3> basket_option{weights, expiry_time, 80};

    double monte_carlo_price = monte_carlo_pricer(model, basket_option);
    double pde_price = pde_pricer(model, basket_option);

    std::cout << "Basket option 2: " << basket_option << "\n" <<
        "\tMonte Carlo price: " << monte_carlo_price << "\n" <<
        "\tPDE price:         " << pde_price << std::endl;
  }

  return 0;
}
