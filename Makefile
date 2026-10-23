# === USER PARAMETERS

ifdef OS
   export UV_INSTALL_CMD=powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
else
   export UV_INSTALL_CMD=curl -LsSf https://astral.sh/uv/install.sh | sh
endif

export SRC_DIR=pricing-advertising
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
MLFLOW_PORT ?= 5000

# == SETUP REPOSITORY AND DEPENDENCIES

install-uv:
	# install uv package manager
	$(UV_INSTALL_CMD)

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
	uv run ruff format

lint:
	uv run ruff check --fix
	uv run mypy --ignore-missing-imports --install-types --non-interactive --package $(SRC_DIR)

test:
	uv run pytest --verbose --color=yes tests

all-validation: format lint test

# === MLFLOW

mlflow-ui:
	uv run mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI) --port $(MLFLOW_PORT)
