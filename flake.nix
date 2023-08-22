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
         name = "python";
         runScript = "zsh";
         targetPkgs = pkgs: [
           pkgs.python311
           pkgs.python311.pkgs.pip
           pkgs.python311.pkgs.virtualenv

           pkgs.poetry

           pkgs.zlib # numpy
           pkgs.geos # cartopy

           # utils
           pkgs.black
         ];
       }).env;
     });
}
