# Project & repository admin notes
Last updated: 2023-12-05

## Hugging Face data
We store some training data and trained models on Hugging Face. They're on
separate repositories under the [M2LInES](https://huggingface.co/M2LInES)
organization. You need org permissions to edit these repositories -- check with
M2LInES folks.

## PyPI library
*We don't publish a PyPI package.* Much of the work is in place, however.

## `flake.nix`, `flake.lock`
These are files to assist building using the Nix package manager. You may ignore
them if you don't use them. They remain useful for certain users, so please do
not delete without consideration.
