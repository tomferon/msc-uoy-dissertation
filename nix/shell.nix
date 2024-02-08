{ pkgs }:

with pkgs;
pkgs.stdenvNoCC.mkDerivation (rec {
  name = "shell";
  buildInputs = with pkgs; [
    texlive.combined.scheme-full
  ];
})
