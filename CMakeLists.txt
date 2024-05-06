cmake_minimum_required(VERSION 3.27)

project(dissertation)

set(CXX_STANDARD 20)
set(CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost 1.84.0
  REQUIRED COMPONENTS ublas)
  # COMPONENTS ublas)

add_executable(dissertation src/main.cpp)

