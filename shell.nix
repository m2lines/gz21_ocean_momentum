# This file is to help Nix users develop the package.
# With Nix installed, run the following commands in this directory:
#
# $ nix-shell
# $ virtualenv venv
# $ source venv/bin/activate
#
# Now you can use `pip install -e .` and generally do regular Python development
# things.
#
# Note that we can't make this a flake easily because of FHS.

{ pkgs ? import <nixpkgs> {} }:

(pkgs.buildFHSUserEnv {
  name = "gz21-ocean-momentum-cnn-devshell";
  runScript = "zsh";
  targetPkgs = pkgs: (with pkgs; [
    pkgs.python3
    pkgs.python3Packages.pip
    pkgs.python3Packages.virtualenv

    pkgs.zlib # numpy
  ]);
}).env
