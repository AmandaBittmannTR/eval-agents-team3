# Contributing to aieng-template

Thanks for your interest in contributing to the aieng-template-implementation!

To submit PRs, please fill out the PR template along with the PR. If the PR
fixes an issue, don't forget to link the PR to the issue!

## Pre-commit hooks

Once the python virtual environment is setup, you can run pre-commit hooks using:

```bash
pre-commit run --all-files
```

## Coding guidelines

For code style, we recommend the [PEP 8 style guide](https://peps.python.org/pep-0008/).

For docstrings we use [numpy format](https://numpydoc.readthedocs.io/en/latest/format.html).

Pre-commit runs hooks such as trailing-whitespace cleanup, YAML/TOML checks, and
[mypy](https://mypy.readthedocs.io/en/stable/) for type checking (see
`.pre-commit-config.yaml`). Use type hints in new code so mypy stays useful.
