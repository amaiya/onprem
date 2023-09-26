# Contributing to OnPrem.LLM

We are happy to accept your contributions to make `onprem` better! To avoid unnecessary work, please stick to the following process:

1. Check if there is already [an issue](https://github.com/amaiya/onprem/issues) for your concern.
2. If there is not, open a new one to start a discussion. We hate to close finished PRs.
3. We would be happy to accept a pull request, if it is decided that your concern requires a code change.


## Developing locally

This project was developed with [nbdev](https://github.com/fastai/nbdev).  All code with the exception of `webapp.py` and `console.py` are maintained in notebooks under the `nbs`.  Before contributing, we recommend going through tne [nbdeb tutorial](https://nbdev.fast.ai/tutorials/tutorial.html).

We suggest cloning the repository and then checking out the documentation and notebooks under `nbs` for information on how to call various methods. Documentation is auto-generated from the notebooks.

See the [installation instructions](https://amaiya.github.io/onprem/#install) for information on setting things up. Using a virtual environment (e.g., in `conda` or `mamba`) is strongly recommended.


## PR Guidelines

- Keep each PR focused. While it's more convenient, please try to avoid combining several unrelated fixes together.
- Try to maintain backwards compatibility.  If this is not possible, please discuss with maintainer(s).
- Use four spaces for indentation.
