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
    texlive.combined.scheme-full
  ];
})
