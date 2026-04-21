# Reinforcement Learning

`src/reinforcement_learning.c` / `include/reinforcement_learning.h` provide a
reactive reward-based RL heuristic used by the main training loop to
nudge spin configurations toward lower energies between neural-network
updates. It is deliberately simple — a placeholder for the true RL
pillar that arrives with v0.5+ work on p-bit / skyrmion-LIF
neuromorphic substrates (see `architecture_v0.4.md` §P2.4).

## 1. Constants

Declared in `include/reinforcement_learning.h`:

```c
#define RL_LEARNING_RATE  0.1    /* flip probability scale factor     */
#define DISCOUNT_FACTOR   0.9    /* reserved for future Q-learning     */
#define INITIAL_EPSILON   0.1    /* ε-greedy exploration, pre-decay   */
#define THRESHOLD         0.01   /* deadband on |ΔE| / |E_prev|       */
```

The module keeps a single file-scope `epsilon` that decays toward
`0.01` as the reward magnitude grows, implementing a crude adaptive
exploration schedule.

## 2. Reward signal

```c
double reward = reinforce_learning(ising_lattice, kitaev_lattice,
                                   current_energy, previous_energy);
```

The reward is computed from the relative drop in total energy:

```
ΔE = E_prev − E_current
raw_reward = max(100 · ΔE / |E_prev|, 0)         [fmax clamp at 0]
reward = raw_reward + random_adjustment          [±0.005 Gaussian-ish jitter]
if |ΔE| < THRESHOLD: reward = 0                  [deadband]
```

Where:

- **Positive reward** iff the current energy is lower than the previous
  (the agent prefers energy minimization).
- **Zero reward** is clamped by `fmax` — energy increases do not
  produce negative reward in this implementation.
- **Small jitter** (`random_adjustment` ∈ [−0.005, +0.005]) prevents
  deterministic oscillation and lets `reward` drift slightly negative
  at the deadband boundary.

As a side effect, the function prints a one-line status to `stdout`:

```
Current Energy: -10.000, Previous Energy: -5.000, Total Energy: 27.000,
Epsilon: 0.037, Random Adjustment: -0.005, Reward: 99.995
```

Failure mode: when `|E_prev| < 10⁻¹⁰` the function returns `0.0` and
logs a diagnostic to `stderr` ("Previous energy too close to zero").
This is the reason the main loop seeds with a warmup iteration.

## 3. Spin optimisation

```c
optimize_spins_with_rl(ising_lattice, kitaev_lattice, reward);
```

Visits every site of both lattices and:

1. If `reward > 0.5` and `|reward| > THRESHOLD`, runs
   `should_flip_spin` on the site. `should_flip_spin` returns `1`
   with probability `reward · RL_LEARNING_RATE` (clamped to `[0, 1]`
   by the `rand()` / `RAND_MAX` uniform draw).
2. Otherwise, flips the site with probability `ε` (ε-greedy
   exploration).

ε itself is updated inside `reinforce_learning`:

```
ε = max(INITIAL_EPSILON · exp(−0.01 · |normalized_reward|), 0.01)
```

so larger rewards drive ε toward the `0.01` floor — more greedy
behaviour as the optimiser becomes confident.

## 4. State strings

```c
char *s_i = get_ising_state_string(ising_lattice);  /* caller must free */
char *s_k = get_kitaev_state_string(kitaev_lattice);
```

Each returns a newline-separated block of `'0'` / `'1'` characters
encoding the per-site spin, sized `size_x * size_y * size_z + size_x`
bytes (one row per `z`-slice with trailing newlines). Useful for
verbose logging from the training loop. Returns `NULL` on allocation
failure; otherwise the caller owns the buffer and must `free()` it.

## 5. Integration in `main.c`

Each training iteration calls:

```c
double reward = reinforce_learning(ising_lattice, kitaev_lattice,
                                   total_energy, previous_energy);
optimize_spins_with_rl(ising_lattice, kitaev_lattice, reward);
previous_energy = total_energy;
```

immediately after the neural network's forward pass and physics-loss
computation. The RL updates therefore act as a between-training-step
state perturber, similar to a simulated-annealing schedule but driven
by the neural network's predicted energy rather than a fixed cooling
curve.

## 6. Tests

`tests/test_reinforcement_learning.c` (4 tests):

- Reward is ≥ 0 when the current energy is lower than the previous
  (the common "good step" case).
- Reward stays near zero (within ±0.01, accounting for jitter) when
  the current energy is higher than the previous.
- `optimize_spins_with_rl` preserves the spin magnitude (± 1) on both
  Ising and Kitaev lattices.
- `get_ising_state_string` / `get_kitaev_state_string` return non-NULL
  and are free-able without crash.

## 7. Known limitations

- **No value function / Q-table.** This is a reactive heuristic, not
  true RL. No state-action pairs are maintained across iterations.
- **One-sided reward.** Energy increases are clamped to zero reward;
  the agent does not directly penalise bad steps.
- **Coupling via `printf`.** The module writes to `stdout` on every
  call; downstream quiet-mode requirements should wrap the call or
  redirect.

These limitations motivate the v0.5+ neuromorphic pillar (P2.4) — see
`architecture_v0.4.md` — which replaces the heuristic with a genuine
probabilistic-bit substrate.
