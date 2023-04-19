{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = inputs@{ self, nixpkgs, flake-utils, ... }:
   flake-utils.lib.eachDefaultSystem (system:
     let
       pkgs = import nixpkgs {
         inherit system overlays;
         config = { allowUnfree = true; };
       };

       overlay = (final: prev: { });
       overlays = [ overlay ];
     in
     rec {
       inherit overlay overlays;
       devShell = (pkgs.buildFHSUserEnv {
         name = "poetry-env";
         runScript = "zsh";
         targetPkgs = pkgs: [
           pkgs.python3
           pkgs.python3Packages.pip
           pkgs.python3Packages.virtualenv

           pkgs.poetry

           pkgs.zlib # numpy
         ];
       }).env;
     });
}
