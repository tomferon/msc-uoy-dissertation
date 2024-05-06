{ pkgs }:

with pkgs;
pkgs.clangStdenv.mkDerivation (rec {
  name = "shell";
  buildInputs = with pkgs; [
    boost185
    clang
    cmake
    cmake-language-server
    texlive.combined.scheme-full
  ];
})
