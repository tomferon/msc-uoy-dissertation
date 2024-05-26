#pragma once

#include <algorithm>
#include <cassert>
#include <iterator>
#include <vector>

#include "model.h"

namespace PDE {

// Coordinates (k_1, ..., k_d) to point to the node
// \hat{C}(T, w_1^* + k_1 h_{w_1}, ..., w_d^* + k_d h_{w_d})
//
// This allows us to have (0,0,...0) as w^* and to work with increments of 1
// instead of h_{w_i}.
template <int D>
class Coordinates {
public:
  const int& operator()(const int i) const {
    return coords[i];
  }

  int& operator()(const int i) {
    return coords[i];
  }

  // Convert (k_1, ..., k_d) to (w_1^* + k_1 h_{w_1}, ..., w_d^* + k_d h_{w_d}).
  ublas::c_vector<double, D> to_weights(
      const ublas::c_vector<double, D>& base_weights, // \mathbf{w}^*
      const ublas::c_vector<double, D>& weight_steps) {

    ublas::c_vector<double, D> coords_vector{};
    for (auto i : std::views::iota(0, D)) {
      coords_vector(i) = coords[i];
    }

    return base_weights + ublas::element_prod(coords_vector, weight_steps);
  }

  // Return new coordinates (k_1, ..., k_i + 1, ..., k_d) given i.
  Coordinates<D> inc(int i) {
    Coordinates<D> new_coords = *this;
    ++new_coords(i);
    return new_coords;
  }

  // Return new coordinates (k_1, ..., k_i - 1, ..., k_d) given i.
  Coordinates<D> dec(int i) {
    Coordinates<D> new_coords = *this;
    --new_coords(i);
    return new_coords;
  }

private:
  std::array<int, D> coords{};
};

template <int D>
std::ostream& operator<<(std::ostream& os, const Coordinates<D>& coords) {
  os << "(" << coords(0);
  for (auto i : std::views::iota(1, D)) {
    os << ", " << coords(i);
  }
  os << ")";
  return os;
}

// All nodes for a fixed expiry time T, required to calculate the values of the
// next layer until reaching \hat{C}(T^*, \mathbf{w}^*).
template <int D>
class TreeLayer {
public:
  // The span of a layer of the tree is k such that the layer has nodes for the
  // weights w_i - k h_{w_i}, ..., w_i + k h_{w_i}. That means there are 1+2*k
  // values for each weights and there are D weights, so the vector of node
  // values will contain (1+2*span)^D values, which we reserve straightaway.
  TreeLayer(int span): span{span}, node_values(pow(1+2*span, D)) {}

  class coordinates_iterator {
  public:
    using iterator_category = std::output_iterator_tag;
    using value_type = Coordinates<D>;
    using difference_type = Coordinates<D>;
    using pointer = Coordinates<D>*;
    using reference = Coordinates<D>&;

    coordinates_iterator(int span, Coordinates<D> coords)
      : span{span}, coords{coords} {}

    bool operator==(const coordinates_iterator& other) const {
      auto range = std::views::iota(0, D);
      return std::all_of(range.begin(), range.end(), [&](auto i) {
        return coords(i) == other.coords(i);
      });
    }

    coordinates_iterator& operator++() {
      for (auto i : std::views::iota(0, D)) {
        coords(i)++;
        if (coords(i) > span && i < D-1) {
          coords(i) = -span;
        } else {
          break;
        }
      }
      return *this;
    }

    Coordinates<D>& operator*() {
      return coords;
    }

  private:
    int span;
    Coordinates<D> coords;
  };

  // We start iterating from (-span, ..., -span) ...
  coordinates_iterator begin() const {
    Coordinates<D> coords{};
    for (auto i : std::views::iota(0, D)) {
      coords(i) = -span;
    }
    return coordinates_iterator(span, coords);
  }

  // ... to (span, ..., span) with (-span, ..., -span, span+1) serving as a
  // sentinel value.
  coordinates_iterator end() const {
    Coordinates<D> coords{};
    for (auto i : std::views::iota(0, D-1)) {
      coords(i) = -span;
    }
    coords(D-1) = span + 1;
    return coordinates_iterator(span, coords);
  }

  // Get the value of the node at the specified coordinate as an array of D
  // integers ranging from -span to span included.
  double& operator()(const Coordinates<D> coords) {
    std::vector<double>::size_type index = 0;
    for (auto d : std::views::iota(0, D)) {
      index *= 1+2*span;
      index += span + coords(d);
    }
    return node_values[index];
  }

  // Return the unique value of this layer if the span is zero. Throw otherwise.
  double value() const {
    assert(span == 0);
    return node_values[0];
  }

private:
  int span;
  std::vector<double> node_values;
};

template <int D>
double pricer(
    const int n, // Number of time steps.
    const Model<D>& model, // Market model.
    const BasketOption<D>& option) { // Specification of basket option to price.

  const double time_step = option.expiry_time / n; // h_T = T^* / n.

  // Values of \hat{C}(T, \mathbf{w}) for a fixed T.
  TreeLayer<D> node_values{n}; // First layer has a span equal to the time steps.

  // Calculate (h_{w_1}, ..., h_{w_d}) based on option.weights and n.
  ublas::c_vector<double, D> weight_steps{};
  for (auto i : std::views::iota(0, D)) {
    weight_steps(i) = option.weights(i) / sqrt(n); // FIXME
  }

  // std::cout << "h_T = " << time_step << "\n";
  // for (auto i : std::views::iota(0, D)) {
  //   std::cout << "h_{w_" << i << "} = " << weight_steps(i) << "\n";
  // }

  // Calculate the values in the first layer using the payoff values.
  for (auto& coords : node_values) {
    ublas::c_vector<double, D> weights =
      coords.to_weights(option.weights, weight_steps);
    double basket_value = ublas::inner_prod(weights, model.initial_prices);
    node_values(coords) = std::max(0.0, basket_value - option.strike_price);
    // std::cout << "C(0, " << coords << ") = " << node_values(coords) << std::endl;
  }

  // Loop over each subsequent layer of the tree until its root.
  for (auto layer : std::views::iota(1, n+1)) {
    std::cout << "\r" << layer << " / " << n << std::flush;
    const double time = time_step * (layer - 1); // T
    const double interest_rate = model.riskfree_rate(time); // r(T)
    const ublas::c_matrix<double, D, D> vol = model.volatility(time); // C(T)

    // Compute A(T) as (C(T) C(T)^T) / 2.
    const ublas::c_matrix<double, D, D> vol_sums =
      ublas::prod(vol, ublas::trans(vol)) / 2.0;

    // std::cout << "C(" << time << ") = {";
    // for (auto i : std::views::iota(0, D)) {
    //   std::cout << "{";
    //   for (auto j : std::views::iota(0, D)) {
    //     std::cout << vol(i, j);
    //     if (j != D-1) std::cout << ", ";
    //   }
    //   std::cout << "}";
    //   if (i != D-1) std::cout << ", ";
    // }
    // std::cout << std::endl;

    // std::cout << "A(" << time << ") = {";
    // for (auto i : std::views::iota(0, D)) {
    //   std::cout << "{";
    //   for (auto j : std::views::iota(0, D)) {
    //     std::cout << vol_sums(i, j);
    //     if (j != D-1) std::cout << ", ";
    //   }
    //   std::cout << "}";
    //   if (i != D-1) std::cout << ", ";
    // }
    // std::cout << std::endl;

    const int span = n - layer;
    // new_node_values corresponds to \hat{C}(T + h_T, \cdot) while node_values
    // corresponds to \hat{C}(T, \cdot).
    TreeLayer<D> new_node_values(span);

    // Compute the node's value using the recursive formula.
// #pragma omp parallel for
    for (auto coords : new_node_values) {
      // \mathbf{w}
      ublas::c_vector<double, D> weights =
        coords.to_weights(option.weights, weight_steps);
      // // Accumulator to compute \hat{C}(T + h_T, \mathbf{w}).
      // double value = node_values(coords);

      double first_term = node_values(coords);
      double second_term = 0.0;
      double third_term = 0.0;

      for (auto i : std::views::iota(0, D)) {
        double theta_i;
        if (coords(i) > -span && coords(i) < span) {
          theta_i =
            ( - node_values(coords.inc(i).inc(i))
              + 8 * node_values(coords.inc(i))
              - 8 * node_values(coords.dec(i))
              + node_values(coords.dec(i).dec(i)) )
            / (12 * weight_steps(i));
        } else {
          theta_i =
            (node_values(coords.inc(i)) - node_values(coords.dec(i)))
            / (2.0 * weight_steps(i));
        }
        second_term += time_step * interest_rate * weights(i) * theta_i;

        // second_term += (time_step / 2.0 / weight_steps(i))
        //   * (node_values(coords.inc(i)) - node_values(coords.dec(i)))
        //   * interest_rate * weights(i);

        for (auto l : std::views::iota(0, D)) {
          // double phi_il =
          //   ( node_values(coords.inc(i).inc(l))
          //     - node_values(coords.inc(i))
          //     - node_values(coords.inc(l))
          //     + node_values(coords) )
          //   / (weight_steps(i) * weight_steps(l));
          double phi_il;
          if (i == l) {
            phi_il = (node_values(coords.inc(i)) - 2 * node_values(coords) + node_values(coords.dec(i))) / pow(weight_steps(i), 2);
          } else {
            phi_il = (node_values(coords.inc(i).inc(l)) - node_values(coords.inc(i).dec(l)) - node_values(coords.dec(i).inc(l)) + node_values(coords.dec(i).dec(l))) / (4 * weight_steps(i) * weight_steps(l));
          }
          // std::cout << "phi_" << i << l << " = " << phi_il << std::endl;

          third_term += weights(i) * weights(l) * phi_il * vol_sums(i, l) * time_step / 2.0;
        }
      }

      // new_node_values(coords) = std::max(-1000.0, std::min(1000.0, first_term + second_term + third_term));
      new_node_values(coords) = first_term + second_term + third_term;
      // std::cout << "C(" << layer << ", " << coords << ") = " << first_term << " + " << second_term << " + " << third_term << " = " << new_node_values(coords) << std::endl;
    }

    node_values = new_node_values;
  }
  std::cout << "                   \r" << std::flush;

  return node_values.value();
}

}
