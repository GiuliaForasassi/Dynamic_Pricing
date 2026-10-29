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

### Warmup

`warmup_days` (60 in `pricing_advertising/__main__.py`, 0 disables it) simulates that many
days before `start_simulation` at every environment reset, so the hotel is not empty on the
first day. Warmup days are not part of the episode: no steps, no reward, no loss, no plots,
and day indices and axes still start at `start_simulation`. Only occupancy, demand and BAR
levels of the nights from `start_simulation` on are kept.

- The warmup actions come from a frozen copy of the current policy: training environments
	refresh it at every rollout start, evaluation environments at every evaluation
	(deterministic). Episodes that end mid-rollout reset with the policy of that rollout.
- Baselines in the PPO reference lines use their own policy during warmup.
- Request calendars include the warmup days, so a seed's calendar differs from the one
	generated with `warmup_days=0`.
- `PricingEnv` needs `set_warmup_policy(callable)` when `warmup_days > 0` (not needed with
	`bar_update_mode` set). Observations passed to the policy are not normalized.

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