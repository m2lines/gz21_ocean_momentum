# Project & repository admin notes
Last updated: 2023-12-05

## Repository automations
### GitHub Actions CI
Workflows are at `.github/workflows/`. Anyone with repository access may change
these.

The linting jobs are currently failing. You may temporarily disable a workflow
(a set of jobs) without removing the configuration by removing the `on:`
top-level key in the appropriate workflow file.

### Integrations
`Review Notebook App` will rear its head when you include changes to a Jupyter
Notebook in a PR. These are configured from `Integrations -> GitHub Apps` on the
Settings tab. You need *maintainer access* to change these (not just admin).

## Repository permissions
Though the repository is under the `m2lines` GitHub organization, permissions
are granted individually. See relevant M2LInES or ICCS members for help with
write access.

## Hugging Face data
We store some training data and trained models on Hugging Face. They're on
separate repositories under the [M2LInES](https://huggingface.co/M2LInES)
organization. You need permissions on the Hugging Face organization to edit
these repositories -- check with M2LInES folks.

## PyPI library
*We don't publish a PyPI package.* Much of the work is in place, however.

## `flake.nix`, `flake.lock`
These are files to assist building using the Nix package manager. You may ignore
them if you don't use them. They remain useful for certain users, so please do
not delete without consideration.
