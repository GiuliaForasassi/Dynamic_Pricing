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

### Experiment configuration

Each experiment uses a standalone JSON file validated at startup with Pydantic.
The default is `configs/quick.json`, a short run for testing and profiling. The
long preset preserves the original PPO experiment:

```bash
python -m pricing_advertising
python -m pricing_advertising --config configs/long.json
python -m pricing_advertising --config configs/quick.json --check-config
```

Copy either file to create another experiment. Its `run`, `simulation`, `ppo`,
`evaluation`, and `cancellation` sections specify the entire run; files do not
inherit or merge settings. An explicit `--config` path is relative to the current
working directory. `bar_table_path` is relative to the JSON file, while model output
and a relative SQLite tracking URI remain relative to the working directory. An
existing `MLFLOW_TRACKING_URI` environment variable overrides the configured URI.
The parent MLflow run stores the effective configuration as `resolved_config.json`.

Use `run.use_subprocesses: false` to profile training in the main process. The
quick preset keeps warmup, cancellations, alternatives, baseline comparisons and
evaluation plots enabled, but is not intended to produce a trained policy.
Individual categories may have zero rooms; the hotel must have at least one room.
The number of category slots and the 360-day observation window remain fixed.

### PPO evaluation configuration

Set `evaluation.eval_seeds` to select evaluation scenarios. The list must be
nonempty, contain distinct seeds in `[0, 2**32)`, and not overlap with any training
environment seed (`seed + index` for each training run).

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
- `evaluation.evaluate_baselines` controls the baseline reference lines in PPO
	plots. Standalone baseline simulations are separate Python APIs and are not
	selected by the experiment CLI.

Parent and child MLflow runs record `n_eval_envs` and `eval_episodes`.

### Warmup

`simulation.warmup_days` (60 in the long preset, 0 disables it) simulates that many
days before `start_simulation` at every environment reset, so the hotel is not empty on the
first day. Warmup days are not part of the episode: no steps, no reward, no loss,
and day indices and axes still start at `start_simulation`. Stay-day request counts and
booked mean prices include warmup requests and confirmed bookings for nights within the
episode; request-day counts exclude warmup arrivals. Booked mean prices exclude cancelled
bookings, while alternatives by original check-in day include warmup offers. Occupancy, demand and BAR
levels of the nights from `start_simulation` on are kept, along with surviving future
bookings when cancellations are enabled.

- The warmup actions come from a frozen copy of the current policy: training environments
	refresh it at every rollout start, evaluation environments at every evaluation
	(deterministic). Episodes that end mid-rollout reset with the policy of that rollout.
- Baselines in the PPO reference lines use their own policy during warmup.
- Request calendars include the warmup days, so a seed's calendar differs from the one
	generated with `warmup_days=0`.
- `PricingEnv` needs `set_warmup_policy(callable)` when `warmup_days > 0` (not needed with
	`bar_update_mode` set). Observations passed to the policy are not normalized.

### Booking cancellations

Configure the `cancellation` section in the experiment JSON:

```json
"cancellation": {
  "daily_probability": 0.005,
  "price_sensitivity": 1.0,
	"high_season_multiplier": 0.8,
	"timing_strength": 0.3,
	"timing_decay_days": 3.0
}
```

Set `"cancellation": null` to disable cancellations throughout the experiment.
`run_simulation` constructs `CancellationModel(**cancellation_config)` only when the
config is not `None`; an empty dictionary enables the model with its defaults.
The same model is passed to training, evaluation, their warmup environments and
the PPO baseline reference lines. `run_simulation` defaults to `cancellation_config=None`.
Direct `PricingEnv` calls still accept `cancellation_model`, defaulting to `None`.
Setting `daily_probability=0` also disables cancellations.
Child MLflow runs record whether cancellations are enabled and the model parameters.

At confirmation, each booking gets a hidden cancellation date (or no cancellation),
sampled using a daily risk that is slightly higher near booking and near check-in.
The overall probability of cancellation before check-in is unchanged from the fixed
daily-risk model. A cancellation is scheduled only if that date falls strictly after
confirmation and strictly before check-in. Future cancellation dates are diagnostic-only, never part
of policy observations or pricing decisions. Cancellation releases the whole stay
before that day's availability checks and pricing, so rooms can be resold immediately.
No cancellation is possible on booking day, check-in day, or during a stay. Pricing sees
released rooms that day, and the policy sees updated occupancy in its next observation.

The model uses the final confirmed category, nightly prices and stay, including
alternative offers. Let $r$ be the average nightly price divided by the category's
reference price, and $f$ the fraction of booked nights in `HIGH_SEASON`. With reference
daily probability $p_0$, price sensitivity $a$ and high-season odds multiplier $m$:

$$
q = \frac{p_0}{1-p_0}\exp(-a(r-1))m^f,
\qquad p_{\text{cancel}} = \frac{q}{1+q}.
$$

At the defaults this gives a base daily risk of 0.5% at reference price in low season, approximately
0.41% at 20% above reference, and 0.40% at reference price in high season. Reference
prices default to a copy of `LOGISTIC_ACCEPTANCE_AVG_PRICES`, ordered by room category;
override the model's `reference_prices` tuple independently if needed. No acceptance
lead-time or seasonal price adjustment is applied to these references. Later BAR
changes do not change an existing booking's cancellation probability. These parameters
are starting assumptions, not fitted cancellation statistics. A daily probability is
not a lifetime probability: at a constant 0.5%, 30 eligible days imply about 14%
eventual cancellations. Explicit probability endpoints 0 and 1 disable cancellations
or force them on the first eligible day.

For $n$ eligible days and day $t=1,\ldots,n$ after booking, let $s$ be
`timing_strength` and $\tau$ be `timing_decay_days`:

$$
a_t = 1+s\left(e^{-(t-1)/\tau}+e^{-(n-t)/\tau}\right),
\qquad w_t = \frac{n a_t}{\sum_{j=1}^{n}a_j},
\qquad p_t = 1-(1-p_{\text{cancel}})^{w_t}.
$$

The default $s=0.3$ gently increases risk at both ends, with the boost decaying over
roughly three days. Risk is slightly reduced in the middle. Since the weights sum to
$n$, total cancellation probability remains $1-(1-p_{\text{cancel}})^n$. One or two
eligible days retain constant risk. `timing_strength` must be finite and non-negative;
`timing_decay_days` must be finite and positive. Setting `timing_strength=0` restores
the original geometric sampler and its seeded trajectories. Enabling the bias changes
seeded trajectories, but runs remain reproducible with the same settings and seed.

The daily risk is conditional on the booking still being active; late cancellation
counts need not match early counts. Stored `cancellation_probability` values remain
the base $p_{\text{cancel}}$, not the day-specific $p_t$. Preserving total probability
per booking does not preserve episode revenue: timing affects resale opportunities
and whether refunds occur before episode end.

- Main-episode bookings earn their full revenue on confirmation day; a cancellation
	subtracts the original full price from reward on cancellation day.
- Warmup bookings can cancel and release rooms, but their refunds never reduce
	main-episode reward because their original revenue was outside the episode.
- There is no settlement after episode end. Late bookings may therefore earn revenue
	without a later refund being simulated. Discounting also makes a delayed refund
	offset less of the original sale in discounted return; reward timing is unchanged.
- `step_metrics` and evaluation artifacts include `cancellations`, `num_cancelled`,
	`gross_booking_revenue`, `refund_total` and `reward_refund_total`. Events link to
	confirmed artifact rows through episode-local `booking_id` and distinguish warmup
	refunds from reward debits. Reward CSVs and `step_reward` report net revenue.
- Historical confirmations, request-cohort snapshots and booking-day revenue charts
	remain gross booking records. Stay-day booked mean prices exclude cancelled bookings;
	refunds are not negative nightly prices. Actual hotel
	occupancy reflects cancellations; historical demand and BAR levels are not reversed.
- A separate cancellation RNG preserves the existing BAR, alternative and acceptance
	streams. Identical policy/configuration/seed replays the same cancellations, including
	warmup; ordinary unseeded resets continue RNG progress.

#### Cancellation Reporting

- `cancellation_amount/by_cancellation_day/mean_completed_episodes/{train,val}`
	counts cancelled reservations on each one-based simulation day, averaged across
	completed episodes. There is no reservation or room-capacity denominator: a cancelled
	multi-night booking counts once. Warmup-origin cancellations count too, even though
	they do not debit reward. Incomplete episodes are excluded.
- `episodes/{train,val}/episode_<id>_cancellation.csv` stores each episode's raw daily
	counts. `cancellation_amount/by_training_timesteps/{train,val}` reports mean daily
	counts over completed episodes since the previous logging step.
- `cancellation_plots/<price_plot_request_date>/<timesteps>_timesteps/env_<id>/episode_<id>.png`
	contains three category panels. The x-axis is absolute stay day (1 = simulation start),
	from the snapshot date through episode end, matching occupancy plots. Each night
	stacks non-voided reservations and voided reservations (scheduled to cancel later but
	still occupying inventory); their sum equals the total-reservations outline.
- Snapshots are end-of-day, after cancellations and new confirmations. Counts include
	all active reservations, including earlier confirmations and warmup bookings, not
	just requests received on the snapshot date. Already-cancelled bookings are excluded.
	Unlike the daily cancellation metric, a multi-night booking contributes once to each
	reserved stay night. Disabled cancellations produce zero voided counts.
- Parent runs write `mean_over_seeds.png` under the same date/timestep folders, averaging
	episodes within each seed before averaging seeds equally. Legacy snapshots without
	voided-state data are not interpreted as zeros; cancellation plots are skipped if
	the necessary data is missing from any contributing snapshot.

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