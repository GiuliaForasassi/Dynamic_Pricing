# === USER PARAMETERS

ifdef OS
   export PYTHON_COMMAND=python
   export UV_INSTALL_CMD=powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   export VENV_BIN=.venv/Scripts
else
   export PYTHON_COMMAND=python3.12
   export UV_INSTALL_CMD=curl -LsSf https://astral.sh/uv/install.sh | sh
   export VENV_BIN=.venv/bin
endif

export SRC_DIR=pricing-advertising
ifndef BRANCH_NAME
	export BRANCH_NAME=$(shell git rev-parse --abbrev-ref HEAD)
endif

# == SETUP REPOSITORY AND DEPENDENCIES

install-uv:
	# install uv package manager
	$(UV_INSTALL_CMD)
	# create environment
	uv venv -p $(PYTHON_COMMAND)


set-hooks:
	cp .hooks/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
	cp .hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
	cp .hooks/post-merge .git/hooks/post-merge && chmod +x .git/hooks/post-merge

# Create or update the virtual environment.
# The project is relocked before syncing.
# This installs all extras and the development group (excluding the build group)
dev-sync:
	uv sync --cache-dir .uv_cache --all-extras


setup: set-hooks dev-sync

# === CODE VALIDATION

format:
	. $(VENV_BIN)/activate && ruff format $(SRC_DIR)

lint:
	. $(VENV_BIN)/activate && ruff check $(SRC_DIR) --fix
	. $(VENV_BIN)/activate && mypy --ignore-missing-imports --install-types --non-interactive --package $(SRC_DIR)

test:
	. $(VENV_BIN)/activate && pytest --verbose --color=yes --cov=$(SRC_DIR) -n auto

all-validation: format lint test