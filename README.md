# How to run

The following commands should be run in the directory of this README file.

```sh
# Generate build files.
cmake -S . -B build
# Compile the program.
cmake --build build --target dissertation
# Run it.
build/dissertation
```

## Dependencies

The code was developed using Clang 16.0.6 on macOS 14.5 with CMake 3.29.2. It uses the following libraries:

* Boost (required): The pricers use Boost uBLAS for vectors and matrices. It expects version 1.85.0.
* OpenMP (optional): If available, OpenMP is used to parallelise the execution of both pricers. It expects version 5.0.

## Jupyter notebook

The notebook uses bindings to the C++ code in `src/python.cpp`. To build them, run

```sh
cmake --build build --target pydissertation
```

Adding `build/` to your `PYTHONPATH` environment variable should make the module `pydissertation` available to Jupyter.
