# Python Service Repository Template
This repository is a template for a Python service repository.

Note: the following guide is based on the latest version of `uv`, the package manager we use, namely the `0.5.1`.
We strongly recommend to always use the latest version of `uv`. Compatibility should always be guaranteed.

## UV installation

You can install uv in 2 ways:
- as a standalone application (https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) (recommended)
- as a python package (https://docs.astral.sh/uv/getting-started/installation/#pypi)

We strongly recommend the **first** option, so that the same version of uv is used all over your projects. In the links above you
will also find information on how to download a specific version.

## pyproject.toml

This file contains the fundamental information about the package.
It is composed of different sections one for each tool or project aspect.

- `[project]`: core of the project info with python version, dependencies and other attributes. When you need to add a project dependency you need to insert a new line in the `dependencies` list. Note that you can specify only the name of the package, a specific version or directly the wheel.
- `[project.optional-dependencies]`: set of optional dependencies that can be installed optionally. They are usually grouped by functionality.
- `[dependency-groups]`: development dependencies. They are local only and are not included in the project requirements when published to PyPI.

### Local dependencies

If you want to use local packages during development you can write this line:
```toml
dependencies = [
    "my-package @ file:/path/to/my/package"
]
```
However, the content of the package will not automatically be updated if it is modified.
To force the refresh, you need to add the parameters `--refresh-package my-package` and `--reinstall-package my-package` in the `uv sync` command.
You can modify the rule `make dev-sync-refresh-package` defined in the Makefile with your libraries
to do this automatically. 

### OS dependent dependencies

If you need to specify OS dependent dependencies you can write something like this:

```toml
dependencies = [
    "torch @ https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp38-cp38-win_amd64.whl ; platform_system == 'Windows'",
    "torch @ https://download.pytorch.org/whl/cpu/torch-2.2.0-cp38-none-macosx_10_9_x86_64.whl ; platform_system == 'Darwin'",
    "torch @ https://download.pytorch.org/whl/cpu/torch-2.2.0%2Bcpu-cp38-cp38-linux_x86_64.whl ; platform_system == 'Linux'",
]
```


## uv.lock


The `uv.lock` file records the exact versions of dependencies installed, along with their sources. This file is universal and cross-platform, capturing the packages
that would be installed across all possible python markers, such as operating systems, architectures and python versions. 

It contains the *exact* resolved versions 
that are installed in the project environment (differently from the `pyproject.toml` which specifies the broad requirements of the project)

The `uv.lock` file is created and updated automatically during operations like `uv sync`, though it can also be manually refreshed using the `uv lock` command. 
While the file is human-readable, it is managed exclusively by uv and should not be edited manually, as it follows a proprietary format not compatible with other tools.

When syncing the environment, uv prioritizes the locked dependency versions over those defined in the pyproject.toml (provided that the latter
are compatible with the locked ones). This allows for reproducible builds across different machines and environments.

## ruff.toml

The `ruff.toml` file contains the configuration for `ruff`, the linter we use.


## Makefile

The Makefile is your best friend, it contains utility commands grouped by batches of instruction executed directly with one command keyword.
Since bash commands depend on operating system, it usually contains if blocks that define the right command to use.

Make file is structured in blocks that organize the rules according to their usage:

- user parameters
- setup repo and dependencies
- code validation
- build and deployment

### USER PARAMETERS

This section contains parameters specific to the repo that are used by the rest of the rules.
The `OS` branch is for Windows OS while the else is for Linux and MacOs.

They are:

- `VENV_BIN`: path of the virtual environment activation scripts
- `PYPI_REPO_URL`: url of the Package repository to push new library packages. It must not contain user and password fields
- `PYPI_INDEX_URL`: url of the PyPi index repository. Note that it contains user and password for our nexus server.
- `SRC_DIR`: source directory of our library package
- `DEPLOY_ENVIRONMENT`: automatically set by reading the current git branch, it is used by the deployment rules
- `USE_DEPLOY_ENVIRONMENT`: if 'y' then add `DEPLOY_ENVIRONMENT` suffix in the version tag


### SETUP REPOSITORY AND DEPENDENCIES

This section contains rules to setup the repo at the first time or to install and update dependencies.

- `setup`: this rule should be called after the first clone of the repo. It sets the hooks and installs the dependencies.
- `dev-sync`: this rule is used to install the dependencies listed in the `pyproject.toml` file. It also creates the environment if not present. While compiling, it creates the `uv.lock` file.

### CODE VALIDATION

This section contains rules for code validation like format, linter and test.
The rule `all-validation` groups all the validation to do on your repo.

### BUILD AND DEPLOY

This section is used to create new package versions.
In particular, the key rule is `deploy-tag` that reads the current version tag `vX.Y.Z@DEPLOY_ENV`.
This rule has the parameter `KIND=major,minor,path` that increments the version number.
Then it updates the `pyproject.toml` and the `__init__.py` files with the new version number and pushes those update with the bump commit.
Note that these operations will trigger the `pre-commit` and `pre-push` hooks that runs validation rules.
Finally, it creates the new tag on the last commit.
To this tag is associated a GitHub action that will trigger the docker build using the rule `docker`.
This rule requires environment variables that are set at GitHub action level.

## Git Hooks

Git hooks are your friends, they help you writing good quality code avoiding mistakes and bugs detecting them before pushing the code on the cloud.
There are three hooks in the repository:

- `pre-commit`: runs the rule `format` to format the code with Python guidelines
- `pre-push`: runs the rules `lint` and `test` to analyze the code and to verify that the tests are not broken
- `post-merge`: runs the rule `dev-sync` to synchronize Python dependencies with the actual `requirements.txt` file

## GitHub Action

GitHub actions is one of the possibile CI/CD solutions, they are procedures that are executed by Github and are used to build docker images.
In this repo there is one workflow named docker` that is triggered by a tag with the pattern `vX.Y.Z@DEPLOY_ENV`.
It is composed of two steps: validation and docker build.
In particular, the validation job executes our custom validation action while the docker build job build the docker image and pushes it to ECR repository.

## Making the template your own

This repository is a template for your projects, when you create a new repository you can select it as template.
After you cloned your repository locally you need to rename entities from `ml3_repo_template` with your project name:

- root service directory
- Makefile `SRC_DIR` variable