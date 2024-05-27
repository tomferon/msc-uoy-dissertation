#pragma once

#include <algorithm>
#include <random>
#include <ranges>

#include "model.h"

namespace MonteCarlo {

template <int D>
ublas::c_vector<double, D> discounted_simulation(
    std::mt19937&, const int, const Model<D>&, const double);

template <int D>
double discounting_factor(const int, const Model<D>&, const double);

// Run the given number of simulations of the prices of the risky assets at the
// expiry time of the option, compute the payoff for each simulation and return
// the average price as the average discounted payoff.
template <int D>
double pricer(
    const unsigned int seed, // Seed for number generation.
    const int m, // Number of simulations to run.
    const int n, // Number of time steps.
    const Model<D>& model, // Market model.
    const BasketOption<D>& option) { // Specification of basket option to price.

  std::mutex total_mutex{};
  double total = 0.0; // Accumulator for sums of all simulations' results.
  int done = 0; // Number of simulations completed.

  // Estimate discounting factor by \hat{D}.
  double df = discounting_factor(n, model, option.expiry_time);

  #pragma omp parallel for
  for (auto i : std::views::iota(0, m)) {
    // Pseudo-random number generator. Each loop iteration gets its own
    // generator with a deterministic seed such that the whole pricing process
    // produces the same result between each run of the program.
    std::mt19937 generator{seed+i};
    // Simulate a realisation of the random vector S(T).
    const ublas::c_vector<double, D> discounted_prices =
      discounted_simulation(generator, n, model, option.expiry_time);
    // B = w^T S(T) = w \cdot (\widetilde{S}(T) / D)
    double basket_value = ublas::inner_prod(option.weights, discounted_prices / df);
    // Add the payoff of the basket option for this simulation to the total.
    {
      std::lock_guard<std::mutex> lock{total_mutex};
      total += std::max(0.0, basket_value - option.strike_price);
      ++done;
      double progress = static_cast<double>(done) / m;
      std::cout << "\r" << round(100 * progress) << "% " << std::flush;
    }
  }
  std::cout << "\r    \r" << std::flush;

  // \hat{C} = \hat{D} \sum_{k=1}^n (w^T \hat{S}(T))^+
  return df * total / m;
}

// Generate a single realisation of the path taken by the discounted price of
// each risky asset and return a vector of the discounted prices at the given
// end time.
template <int D>
ublas::c_vector<double, D> discounted_simulation(
    std::mt19937& generator, // Pseudo-random number generator.
    const int n, // Number of time steps.
    const Model<D>& model,
    const double end_time) {

  // Standard normal distribution we draw from to generate Wiener increments.
  std::normal_distribution<double> stdnorm{0.0, 1.0};
  // Exponents s_i such that S_i(T) = S_i(0) * D * exp(e_i).
  ublas::c_vector<double, D> s{ublas::zero_vector(D)};

  // At each time step, we use the distribution above to simulate
  // W((k+1)h) - W(kh).
  ublas::c_vector<double, D> wiener_increments{};

  const double h = end_time / n;

  ublas::c_vector<double, D> hs{}; // [h h ... h]^T
  for (auto i : std::views::iota(0, D)) {
    hs(i) = h;
  }

  for (auto k : std::views::iota(1, n+1)) {
    const double time = k * h; // Time of this step.

    // Generate Wiener increments \Delta_k for this step.
    for (auto j : std::views::iota(0, D)) { // For each Wiener process.
      wiener_increments(j) = stdnorm(generator) * std::sqrt(h);
    }

    const ublas::c_matrix<double, D, D> Ckh = model.volatility(time); // C(kh)

    s -= 0.5 * ublas::prod(ublas::element_prod(Ckh, Ckh), hs);
    // s += (-0.5) * ublas::prod(Ckh, trans(Ckh)) * h;
    s += ublas::prod(Ckh, wiener_increments);
  }

  // Compute the prices of the risky assets at the end time from the exponents.
  ublas::c_vector<double, D> discounted_prices{}; // \widetilde{S}(T)
  for (auto i : std::views::iota(0, D)) {
    discounted_prices(i) = model.initial_prices(i) * exp(s(i));
  }
  return discounted_prices;
}

// Estimate the discounting factor using the lower Riemann sum as described in
// the PDF.
template <int D>
double discounting_factor(
    const int n, // Number of time steps.
    const Model<D>& model,
    const double end_time) {

  double sum = 0; // \sum_{k=1}^n r(kh) h
  double h = end_time / n;

  for (auto k : std::views::iota(0, n)) {
    sum += model.riskfree_rate(k * h) * h;
  }

  return exp(-sum);
}

}
