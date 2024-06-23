{ pkgs }:

with pkgs;

let
  python = pkgs.python311.withPackages (ps: with ps; [
    jupyter
    numpy
    pandas
    pybind11
    scipy
    seaborn
  ]);

in
pkgs.clangStdenv.mkDerivation (rec {
  name = "shell";
  buildInputs = [
      python
    ] ++ (with pkgs; [
      boost185
      clang
      cmake
      cmake-language-server
      llvmPackages.openmp
      texlive.combined.scheme-full
    ]);
})
