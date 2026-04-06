# 2D Coulomb Plasma -- Metropolis Monte Carlo

Simulation of 2D Coulomb gas / plasma model for studying BKT phase transition using Monte Carlo (Metropolis) methods.

## Metropolis algorithm for the 2D many-body plasma

We use the same logic as in the Ising-model Metropolis algorithm, except that the microscopic state is a configuration of particle positions rather than a spin configuration.

### Microstate and Boltzmann weight

Take $N$ particles in $2$D, with fixed charges

$$
s_i \in \{+1,-1\},
\qquad
\sum_{i=1}^N s_i = 0,
$$

and positions

$$
\mathbf X = (\mathbf r_1,\mathbf r_2,\dots,\mathbf r_N).
$$

Here $\mathbf X$ is the microstate of the system.

Using only the first-term interaction, the energy of a configuration is

$$
\boxed{
H(\mathbf X)
=
- K \sum_{i<j} s_i s_j \ln\!\left(\frac{|\mathbf r_i-\mathbf r_j|}{\xi}\right)
}
$$

together with the hard-core constraint

$$
|\mathbf r_i-\mathbf r_j| \ge a
\qquad \text{for all } i \neq j.
$$

The probability density of finding the system in configuration $\mathbf X$ in the canonical ensemble is

$$
\boxed{
\pi(\mathbf X)
=
\frac{e^{-\beta H(\mathbf X)}}{Z}
=
\frac{1}{Z} e^{-\beta H(\mathbf X)}
}
$$

with

$$
\beta = \frac{1}{T}.
$$

Since the positions are continuous, $\pi(\mathbf X)$ is a probability density on configuration space.

### The Metropolis proposal

The analog of a spin flip is now a single-particle displacement.

Starting from a configuration $\mathbf X$,

1. randomly choose one particle $i$,
2. propose moving it from $\mathbf r_i$ to a nearby point $\mathbf r_i'$,
3. this generates a new configuration $\mathbf X'$.

So the proposal is

$$
\mathbf X = (\mathbf r_1,\dots,\mathbf r_i,\dots,\mathbf r_N)
\;\longrightarrow\;
\mathbf X' = (\mathbf r_1,\dots,\mathbf r_i',\dots,\mathbf r_N).
$$

### Energy change

The proposed move changes the energy by

$$
\Delta E = H(\mathbf X') - H(\mathbf X).
$$

Because only one particle moves, only the terms involving particle $i$ need to be recomputed. Therefore

$$
\boxed{
\Delta E
=
- K \sum_{j\neq i} s_i s_j
\ln\!\left(
\frac{|\mathbf r_i'-\mathbf r_j|}{|\mathbf r_i-\mathbf r_j|}
\right)
}
$$

provided the trial move satisfies the hard-core constraint and stays inside the allowed region.

### Acceptance probability

The Metropolis rule accepts the proposed move with probability

$$
P_{\text{accept}}
=
\min\!\left(1,\frac{\pi(\mathbf X')}{\pi(\mathbf X)}\right).
$$

Using the canonical Boltzmann form,

$$
\frac{\pi(\mathbf X')}{\pi(\mathbf X)}
=
\frac{e^{-\beta H(\mathbf X')}}{e^{-\beta H(\mathbf X)}}
=
e^{-\beta\left(H(\mathbf X')-H(\mathbf X)\right)}
=
e^{-\beta \Delta E}.
$$

Hence the acceptance rule becomes

$$
\boxed{
P_{\text{accept}}
=
\min\!\left(1,e^{-\beta \Delta E}\right)
}
$$

which is the same structure as in the spin system.

### Algorithm

For this plasma model, the Metropolis algorithm is therefore:

1. Propose a move by randomly selecting one particle $i$.
2. Displace that particle by a small random vector, producing a trial configuration $\mathbf X'$.
3. Calculate the energy change
   $$
   \Delta E = H(\mathbf X') - H(\mathbf X).
   $$
4. Accept the move with probability
   $$
   P_{\text{accept}}=\min\!\left(1,e^{-\beta \Delta E}\right).
   $$

If the move is rejected, the configuration remains $\mathbf X$.

### Physical meaning of the acceptance rule

This rule favors lower-energy moves:

- if $\Delta E \le 0$, then $e^{-\beta \Delta E}\ge 1$, so the move is accepted with probability $1$;
- if $\Delta E > 0$, then the move is still accepted with probability $e^{-\beta \Delta E}$.

This allows the Markov chain to explore configuration space while still converging to the equilibrium Boltzmann distribution.

### In this specific model

For the low-$T$ logarithmic plasma, the algorithm samples configurations according to

$$
\pi(\mathbf X)
\propto
\exp\!\left[
\frac{K}{T}\sum_{i<j}s_i s_j
\ln\!\left(\frac{|\mathbf r_i-\mathbf r_j|}{\xi}\right)
\right].
$$

So the Monte Carlo trajectory spends more time in configurations where

- opposite charges are favorably arranged,
- like charges tend to stay apart,
- hard-core overlaps are excluded.

That gives a practical numerical way to study many-body observables such as nearest opposite-charge distance, pair correlations, or cluster structure.


## Files

- `simulation.py` -- MC engine (energy functions, initialization, Metropolis loop)
- `plotting.py` -- animation and visualization
- `lowT_sim.ipynb` -- configure parameters and run interactively

## Parameters

Edit the variables at the top of `lowT_sim.ipynb`:
- `L` -- box side length (default 200)
- `N_plus`, `N_minus` -- number of +/- charges (default 20 each)
- `K` -- coupling constant (default 2.0)
- `a` -- hard-core radius (default 1.0)
- `n_sweeps` -- MC sweeps (default 5000)
