# Dynamic Pricing and Online Advertising

## Usage

The project is based on [UV](https://docs.astral.sh/) as package manager and Make as command runner. You need to have both installed in your system to use this template.

### UV installation

To install `uv` you can use the following command:

```bash
make install-uv
```

Once you have those, you can just run

```bash
make dev-sync
```

### Formatting, Linting and Testing

You can configure Ruff by editing the `.ruff.toml` file. It is currently set to the default configuration.

Format your code:

```bash
make format
```

Run linters (ruff and mypy):

```bash
make lint
```

Run tests:

```bash
make test
```

Or you can run them all at once:

```bash
make all-validation
```