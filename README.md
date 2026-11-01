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
and day indices and axes still start at `start_simulation`. Occupancy, demand and BAR
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

Configure `cancellation_config` in `pricing_advertising/__main__.py`:

```python
cancellation_config = dict(
	daily_probability=0.005,
	price_sensitivity=1.0,
	high_season_multiplier=0.8,
)
```

Set `cancellation_config = None` to disable cancellations throughout the experiment.
`run_simulation` constructs `CancellationModel(**cancellation_config)` only when the
config is not `None`; an empty dictionary enables the model with its defaults.
The same model is passed to training, evaluation, their warmup environments and
the PPO baseline reference lines. `run_simulation` defaults to `cancellation_config=None`.
Direct `PricingEnv` calls still accept `cancellation_model`, defaulting to `None`.
Setting `daily_probability=0` also disables cancellations.
Child MLflow runs record whether cancellations are enabled and the model parameters.

At confirmation, each booking gets a hidden geometric waiting time using its fixed
daily cancellation probability. A cancellation is scheduled only if that date falls
strictly after confirmation and strictly before check-in. This has the same distribution
as independent daily draws, but changes seeded trajectories compared with the earlier
daily-draw implementation. Future cancellation dates are diagnostic-only, never part
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

At the defaults this gives 0.5% daily at reference price in low season, approximately
0.41% at 20% above reference, and 0.40% at reference price in high season. Reference
prices default to a copy of `LOGISTIC_ACCEPTANCE_AVG_PRICES`, ordered by room category;
override the model's `reference_prices` tuple independently if needed. No acceptance
lead-time or seasonal price adjustment is applied to these references. Later BAR
changes do not change an existing booking's cancellation probability. These parameters
are starting assumptions, not fitted cancellation statistics. A daily probability is
not a lifetime probability: at a constant 0.5%, 30 eligible days imply about 14%
eventual cancellations. Explicit probability endpoints 0 and 1 disable cancellations
or force them on the first eligible day.

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
- Historical confirmations, request-cohort snapshots and booked-price/revenue charts
	remain gross booking records. Refunds are not negative nightly prices. Actual hotel
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