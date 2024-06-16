{ pkgs }:

with pkgs;
pkgs.clangStdenv.mkDerivation (rec {
  name = "shell";
  buildInputs = with pkgs; [
    boost185
    clang
    cmake
    cmake-language-server
    llvmPackages.openmp
    python311
    python311Packages.jupyter
    python311Packages.numpy
    python311Packages.pybind11
    python311Packages.seaborn
    texlive.combined.scheme-full
  ];
})
