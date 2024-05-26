#include <iostream>
#include <ranges>

#include "model.h"
#include "monte_carlo.h"
#include "pde.h"

constexpr double expiry_time = 100.0 / 365.0;
constexpr int monte_carlo_simulation_count = 10000;
constexpr int monte_carlo_time_step_count = 1000;
constexpr int pde_time_step_count = 100;

// Seed for the pseudo-random number generator. Set here for reproducibility.
constexpr unsigned int seed = 20240516;

ublas::c_matrix<double, 3, 3> example_volatility_matrix(const double t) {
  ublas::c_matrix<double, 3, 3> coefficients{};
  for (auto i : std::views::iota(0, 3)) {
    for (auto j : std::views::iota(0, 3)) {
      // We introduce some shifts so entries change differently over time.
      int shift1 = i + j % 5 - 2;
      int shift2 = i + j % 4 - 2;
      coefficients(i,j) = 0.01 * std::pow(t - 5 + shift1, 2) + 0.1 * shift2;
    }
  }
  return coefficients;
}

ublas::c_matrix<double, 2, 2> example_volatility_matrix2(const double t) {
  ublas::c_matrix<double, 2, 2> coefficients{};
  // for (auto i : std::views::iota(0, 2)) {
  //   for (auto j : std::views::iota(0, 2)) {
  //     // We introduce some shifts so entries change differently over time.
  //     int shift1 = i + j % 5 - 2;
  //     int shift2 = i + j % 4 - 2;
  //     coefficients(i,j) = 0.01 * std::pow(t - 5 + shift1, 2) + 0.1 * shift2;
  //   }
  // }
  coefficients(0, 0) = 0.1;
  coefficients(1, 0) = 0.1;
  coefficients(0, 1) = 0.0;
  coefficients(1, 1) = 0.0;
  return coefficients;
}

double example_riskfree_rate(const double t) {
  return 0.025; // FIXME + t * 0.002;
}

int main(int argc, char *argv[]) {
  {
    // Example instance of the local volatility model with time-dependent only
    // volatily coefficients.
    ublas::c_vector<double, 2> initial_prices{};
    initial_prices(0) = 100.0;
    initial_prices(1) = 100.0;
    const Model<2> model{
      initial_prices,
      example_volatility_matrix2,
      example_riskfree_rate
    };

    // Example 1
    {
      ublas::c_vector<double, 2> weights{};
      // weights(0) = 2.5;
      // weights(1) = 3.0;
      weights(0) = 0.5;
      weights(1) = 0.5;
      const BasketOption<2> basket_option{weights, expiry_time, 100};

      double monte_carlo_price = MonteCarlo::pricer(
        seed, monte_carlo_simulation_count, monte_carlo_time_step_count,
        model, basket_option);

      std::cout << "Basket option 0: " << basket_option << "\n" <<
        "\tMonte Carlo price: " << monte_carlo_price << "\n";
      for (auto step : std::vector{500, 1000, 1500, 2000}) {
        double pde_price = PDE::pricer(step, model, basket_option);
        std::cout << "\tPDE price with " << step << " steps: " << pde_price << std::endl;
      }
    }
  }

  return 0;

  // Example instance of the local volatility model with time-dependent only
  // volatily coefficients.
  ublas::c_vector<double, 3> initial_prices{};
  initial_prices(0) = 100.0;
  initial_prices(1) = 20.0;
  initial_prices(2) = 50.0;
  const Model<3> model{
    initial_prices,
    example_volatility_matrix,
    example_riskfree_rate
  };

  // Example 1
  {
    ublas::c_vector<double, 3> weights{};
    weights(0) = 2.5;
    weights(1) = 3.0;
    weights(2) = 1.0;
    const BasketOption<3> basket_option{weights, expiry_time, 350.0};

    double monte_carlo_price = MonteCarlo::pricer(
      seed, monte_carlo_simulation_count, monte_carlo_time_step_count,
      model, basket_option);
    double pde_price = PDE::pricer(pde_time_step_count, model, basket_option);

    std::cout << "Basket option 1: " << basket_option << "\n" <<
      "\tMonte Carlo price: " << monte_carlo_price << "\n" <<
      "\tPDE price:         " << pde_price << "\n" << std::endl;
  }

  // Example 2
  {
    ublas::c_vector<double, 3> weights{};
    weights(0) = 4.0;
    weights(1) = 1.5;
    weights(2) = 6.0;
    const BasketOption<3> basket_option{weights, expiry_time, 800.0};

    double monte_carlo_price = MonteCarlo::pricer(
      seed,  monte_carlo_simulation_count, monte_carlo_time_step_count,
      model, basket_option);
    double pde_price = PDE::pricer(pde_time_step_count, model, basket_option);

    std::cout << "Basket option 2: " << basket_option << "\n" <<
      "\tMonte Carlo price: " << monte_carlo_price << "\n" <<
      "\tPDE price:         " << pde_price << std::endl;
  }

  return 0;
}
