# === USER PARAMETERS

ifdef OS
   export VENV_BIN=.venv/Scripts
else
   export VENV_BIN=.venv/bin
endif

export SRC_DIR=pricing-advertising
ifndef BRANCH_NAME
	export BRANCH_NAME=$(shell git rev-parse --abbrev-ref HEAD)
endif
DEPLOY_ENVIRONMENT=$(shell if [ $(findstring main, $(BRANCH_NAME)) ]; then \
			echo 'prod'; \
		elif [ $(findstring pre, $(BRANCH_NAME)) ]; then \
			echo 'pre'; \
		else \
		 	echo 'dev'; \
		fi)
# If use deploy_environment in the tag system
# `y` => yes
# `n` => no
USE_DEPLOY_ENVIRONMENT=n

# == SETUP REPOSITORY AND DEPENDENCIES

eval-extras:
	$(eval EXTRA_FLAGS := $(foreach extra,$(EXTRAS),--extra $(extra)))

set-hooks:
	cp .hooks/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
	cp .hooks/pre-push .git/hooks/pre-push && chmod +x .git/hooks/pre-push
	cp .hooks/post-merge .git/hooks/post-merge && chmod +x .git/hooks/post-merge

# Create or update the virtual environment.
# The project is relocked before syncing.
# This installs all extras and the development group (excluding the build group)
dev-sync: refresh-codeartifact-token
	uv sync --index $(PYPI_INDEX_URL) --cache-dir .uv_cache --all-extras --only-group dev

# Sync environment as in dev-sync but also refreshes a certain package,
# which might be a local version
dev-sync-refresh-package: refresh-codeartifact-token
	uv sync --index $(PYPI_INDEX_URL) --cache-dir .uv_cache --all-extras --only-group dev --refresh-package <YOUR_LIB> --reinstall-package <YOUR_LIB>

# Same as dev-sync but installs only the requested extras.
# Usage: make dev-sync-extras EXTRAS="data"
dev-sync-extras: refresh-codeartifact-token eval-extras
	uv sync --index $(PYPI_INDEX_URL) --cache-dir .uv_cache $(EXTRA_FLAGS)

# Available but not working properly, still needs to be refined
dev-add-local-dep: refresh-codeartifact-token
	uv add --editable "<PATH_TO_YOUR_LIB>" --cache-dir .uv_cache --index $(PYPI_INDEX_URL)

editable-utils-install:
	uv pip install -e <PATH_TO_YOUR_LIB> --cache-dir .uv_cache

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

# === BUILD AND DEPLOYMENT

deploy-tag:
	# This rule reads the current version tag, creates a new one with
	# the increment according to the variable KIND

	@# check if KIND variable is set
	@[ -z "$(KIND)" ] && echo KIND is empty && exit 1 || echo "creating tag $(KIND)"

	@# check if KIND variable has the allowed value
	@if [ "$${KIND}" != "major" -a "$${KIND}" != "minor" -a "$${KIND}" != "patch" ]; then \
		echo "Error: KIND environment variable must be set to 'major', 'minor', 'patch' or 'beta'."; \
		exit 1; \
	fi

	@# read the current tag and export the three kinds
	@# to retrieve the version levels, we separate them by white space
	@# to do that we need to replace . and -
	@# then we keep the words number 1, 2, and 3
ifeq (USE_DEPLOY_ENVIRONMENT, y)
	$(eval CURRENT_TAG=$(shell git describe --tags --abbrev=0 --match="v*@$(DEPLOY_ENVIRONMENT)"))
else
	$(eval CURRENT_TAG=$(shell git describe --tags --abbrev=0 --match="v*"))
endif
	$(eval MAJOR=$(shell echo echo $(CURRENT_TAG) | cut -d '@' -f 1 | cut -d 'v' -f 2 | cut -d '.' -f 1))
	$(eval MINOR=$(shell echo echo $(CURRENT_TAG) | cut -d '@' -f 1 | cut -d 'v' -f 2 | cut -d '.' -f 2))
	$(eval PATCH=$(shell echo echo $(CURRENT_TAG) | cut -d '@' -f 1 | cut -d 'v' -f 2 | cut -d '.' -f 3))
	@echo "Version: $(CURRENT_TAG)"
	@echo "Major: $(MAJOR)"
	@echo "Minor: $(MINOR)"
	@echo "Patch: $(PATCH)"
	$(eval OLD_VERSION=$(MAJOR).$(MINOR).$(PATCH))

	@# according to the kind set the new tag
	@# I know it's strange but if blocks must be written without indentation
ifeq ($(KIND),major)
	$(eval MAJOR := $(shell echo $$(($(MAJOR) + 1))))
	$(eval MINOR := 0)
	$(eval PATCH := 0)
else ifeq ($(KIND),minor)
	$(eval MINOR := $(shell echo $$(($(MINOR) + 1))))
	$(eval PATCH := 0)
else ifeq ($(KIND),patch)
	$(eval PATCH := $(shell echo $$(($(PATCH) + 1))))
endif

	@# we add a prefix to the tag to specify the deploy environment
	$(eval DEPLOY_ENVIRONMENT_SUFFIX = @$(DEPLOY_ENVIRONMENT))

	@# Set the new tag variable
	$(eval NEW_VERSION=$(MAJOR).$(MINOR).$(PATCH))
	$(eval NEW_TAG=v$(NEW_VERSION)$(DEPLOY_ENVIRONMENT_SUFFIX))
	$(eval MESSAGE=new version $(NEW_TAG))

	@# Update pyproject.toml with new version
	@echo "Updating pyproject.toml"
ifdef OS
	sed -i "s/version = \""$(OLD_VERSION)"\"/version = \""$(NEW_VERSION)"\"/" pyproject.toml
	sed -i "s/__version__ = '$(OLD_VERSION)'/__version__ = '$(NEW_VERSION)'/" $(SRC_DIR)/__init__.py
else
	sed -i '' "s/version = \""$(OLD_VERSION)"\"/version = \""$(NEW_VERSION)"\"/" pyproject.toml
	sed -i '' "s/__version__ = '$(OLD_VERSION)'/__version__ = '$(NEW_VERSION)'/" $(SRC_DIR)/__init__.py
endif
	git add pyproject.toml
	git add $(SRC_DIR)/__init__.py
	git commit -m "bump version $(OLD_VERSION)->$(NEW_VERSION)"
	git push

	@echo $(NEW_TAG)
	@# create new tag
	git tag -a $(NEW_TAG) -m "$(MESSAGE)"

	@# push the tag
	@# the push of this tag will trigger the github action that builds the package	
	git push origin $(NEW_TAG) --no-verify

deploy-tag-patch:
	@make deploy-tag KIND=patch

deploy-tag-minor:
	@make deploy-tag KIND=minor

deploy-tag-major:
	@make deploy-tag KIND=major