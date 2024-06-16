cmake_minimum_required(VERSION 3.27)

project(dissertation)

set(CXX_STANDARD 20)
add_compile_options(-std=c++20)
set(CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Fix problems with clangd not finding standard libraries.
# See https://discourse.nixos.org/t/get-clangd-to-find-standard-headers-in-nix-shell/11268/10
if(CMAKE_EXPORT_COMPILE_COMMANDS)
  set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES
      ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
endif()

find_package(Boost 1.85.0 REQUIRED)
find_package(OpenMP 5.0)

add_executable(dissertation src/main.cpp src/model.h src/monte_carlo.h src/pde.h)
if(OpenMP_CXX_FOUND)
    target_link_libraries(dissertation PUBLIC OpenMP::OpenMP_CXX)
endif()

find_package(Python 3.11 COMPONENTS Interpreter Development)
find_package(pybind11)
if(pybind11_FOUND)
    pybind11_add_module(pydissertation MODULE src/python.cpp)
endif()

