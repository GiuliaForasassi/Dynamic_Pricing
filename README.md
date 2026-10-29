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

### PPO evaluation configuration

Edit `eval_seeds` in `pricing_advertising/__main__.py` to select the evaluation
scenarios. It defaults to a copy of `EVAL_SEEDS` from `utils/constants.py`; you can
replace it in the main with an explicit list, for example `eval_seeds = [1000001, 1000002]`.
The list must be nonempty, contain distinct seeds in `[0, 2**32)`, and not overlap
with any training environment seed (`seed + index` for each training run).

- `n_envs` controls parallel **training** environments only.
- `n_eval_envs = len(eval_seeds)` and `n_eval_episodes = n_eval_envs` are derived,
	not independent settings: each PPO evaluation runs one episode per evaluation seed.
- All training runs use the same evaluation seeds. Each evaluation environment
	replays its fixed scenario at every checkpoint and in the final evaluation.
- `eval_frequency` is a positive number of vectorized training steps between
	evaluations, passed directly to the callback. The interval in total training
	timesteps is `eval_frequency * n_envs`, recorded in child MLflow runs as
	`effective_eval_frequency`: `10000` with four training environments means
	one evaluation every 40,000 total training timesteps.
- `baseline_eval_episodes` applies only to the commented-out standalone baseline
	simulations, which use one environment and one `eval_seed`, with random streams
	continuing across repeated episodes. It does not configure PPO or the baseline
	reference lines in PPO plots: those already use one episode per `eval_seeds` entry.

Parent and child MLflow runs record `n_eval_envs` and `eval_episodes`.

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