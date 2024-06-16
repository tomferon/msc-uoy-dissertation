#include <vector>

#include <boost/numeric/ublas/matrix.hpp>
#include <pybind11/cast.h>
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include "model.h"
#include "pde.h"

namespace py = pybind11;

namespace PYBIND11_NAMESPACE { namespace detail {
  template <int D>
  struct type_caster<ublas::c_vector<double, D>> {
    using vector = ublas::c_vector<double, D>;

    PYBIND11_TYPE_CASTER(vector, _("Vector"));

    bool load(handle src, bool) {
      auto list = reinterpret_borrow<py::list>(src);
      if (list.size() != D) {
        return false;
      }
      for (auto i : std::views::iota(0, D)) {
        value(i) = list[i].cast<double>();
      }
      return true;
    }

    static handle cast(vector src, return_value_policy, handle) {
      py::list list{D};
      for (auto i : std::views::iota(0, D)) {
        list[i] = PyFloat_FromDouble(src(i));
      }
      return list.release();
    }
  };

  template <int D>
  struct type_caster<ublas::c_matrix<double, D, D>> {
    using matrix = ublas::c_matrix<double, D, D>;

    PYBIND11_TYPE_CASTER(matrix, _("Matrix"));

    bool load(handle src, bool) {
      auto list = reinterpret_borrow<py::list>(src);
      if (list.size() != D) {
        return false;
      }
      for (auto i : std::views::iota(0, D)) {
        auto sublist = list[i].cast<py::list>();
        if (sublist.size() != D) {
          return false;
        }
        for (auto j : std::views::iota(0, D)) {
          value(i, j) = sublist[j].cast<double>();
        }
      }
      return true;
    }

    static handle cast(matrix src, return_value_policy, handle) {
      py::list list{D};
      for (auto i : std::views::iota(0, D)) {
        py::list sublist{D};
        for (auto j : std::views::iota(0, D)) {
          sublist[j] = PyFloat_FromDouble(src(i, j));
        }
        list[i] = sublist;
      }
      return list.release();
    }
  };
}}

PYBIND11_MODULE(pydissertation, m) {
  m.doc() = "Bindings to use PDE pricer in Python";

  py::class_<Model<2>>(m, "Model")
    .def(py::init<ublas::c_vector<double, 2>, volatility_matrix<2>&, interest_rate&>(),
        py::arg("initial_prices"), py::arg("volatility"), py::arg("riskfree_rate"))
    .def_readwrite("initial_prices", &Model<2>::initial_prices)
    .def_readwrite("volatility", &Model<2>::volatility)
    .def_readwrite("riskfree_rate", &Model<2>::riskfree_rate);

  py::class_<BasketOption<2>>(m, "BasketOption")
    .def(py::init<ublas::c_vector<double, 2>, double, double>(),
        py::arg("weights"), py::arg("expiry_time"), py::arg("strike_price"))
    .def_readwrite("weights", &BasketOption<2>::weights)
    .def_readwrite("expiry_time", &BasketOption<2>::expiry_time)
    .def_readwrite("strike_price", &BasketOption<2>::strike_price);

  m.def("pde_pricer", &PDE::pricer<2>, "Price a basket option using the PDE");
}
