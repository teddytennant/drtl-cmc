"""Section 5.1 data generating process.

Z2, Z3 ~ N(0, AR(1) with correlation 0.2), Z = (1, Z2, Z3).
logit P(Y=1 | Z) = Z' (0.5, 0.8, -0.3).
Mcor: X = 1 + 0.4 Y - 0.6 Z2 + 0.2 Z3 + eps, eps ~ N(0, 0.25).
Mmis: Mcor plus Z2^2.
Wcor: logit P(S=1 | Y, Z) = -0.8 + 0.8 Y - 0.2 Z2 + 0.3 Z3.
Wmis: Wcor minus 0.3 Y Z3.

Rows are reordered so the source (S=1) comes first. X on the target is
returned only as a latent column for benchmarks; estimators must not use it.
"""

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from drtl_cmc._jaxsetup import ensure_x64

ensure_x64()

Array = jnp.ndarray

OUTCOME_COEF = jnp.array([0.5, 0.8, -0.3])
X_COEF_Y = 0.4
X_COEF_Z = jnp.array([-0.6, 0.2])
X_INTERCEPT = 1.0
X_NOISE_SD = 0.5  # variance 0.25 = 0.5**2
SELECTION_INTERCEPT = -0.8
SELECTION_Y = 0.8
SELECTION_Z = jnp.array([-0.2, 0.3])
SELECTION_INTERACTION = -0.3
AR_RHO = 0.2


class CMCSample(NamedTuple):
    y_source: Array
    x_source: Array
    z_source: Array
    y_target: Array
    z_target: Array
    x_target: Array  # latent; masked in the target observation model
    s: Array


def ar1_cov(d: int = 2, rho: float = AR_RHO) -> Array:
    idx = jnp.arange(d)
    return rho ** jnp.abs(idx[:, None] - idx[None, :])


def _mechanism_mean(y: Array, z: Array, mechanism: str) -> Array:
    z2 = z[:, 1]
    z3 = z[:, 2]
    mean = X_INTERCEPT + X_COEF_Y * y + X_COEF_Z[0] * z2 + X_COEF_Z[1] * z3
    if mechanism == "Mmis":
        mean = mean + z2**2
    elif mechanism != "Mcor":
        raise ValueError("mechanism must be 'Mcor' or 'Mmis'")
    return mean


def _selection_logit(y: Array, z: Array, membership: str) -> Array:
    z2 = z[:, 1]
    z3 = z[:, 2]
    logit = (
        SELECTION_INTERCEPT
        + SELECTION_Y * y
        + SELECTION_Z[0] * z2
        + SELECTION_Z[1] * z3
    )
    if membership == "Wmis":
        logit = logit + SELECTION_INTERACTION * y * z3
    elif membership != "Wcor":
        raise ValueError("membership must be 'Wcor' or 'Wmis'")
    return logit


def simulate_complete(
    n: int,
    key: Array,
    mechanism: str = "Mcor",
    membership: str = "Wcor",
) -> tuple:
    """Draw n i.i.d. complete rows from the Section 5.1 super-population."""
    k_z, k_y, k_x, k_s = jax.random.split(key, 4)
    z23 = jax.random.multivariate_normal(
        k_z, jnp.zeros(2), ar1_cov(), (n,), dtype=jnp.float64
    )
    z = jnp.concatenate([jnp.ones((n, 1)), z23], axis=1)
    prob_y = jax.nn.sigmoid(z @ OUTCOME_COEF)
    y = jax.random.bernoulli(k_y, prob_y).astype(jnp.float64)
    x = _mechanism_mean(y, z, mechanism) + X_NOISE_SD * jax.random.normal(
        k_x, (n,), dtype=jnp.float64
    )
    prob_s = jax.nn.sigmoid(_selection_logit(y, z, membership))
    s = jax.random.bernoulli(k_s, prob_s)
    return y, x, z, s


def split_source_target(y: Array, x: Array, z: Array, s: Array) -> CMCSample:
    """Reorder so source rows (S=1) occupy 1..n and target rows the rest."""
    source = s.astype(bool)
    target = ~source
    return CMCSample(
        y_source=y[source],
        x_source=x[source],
        z_source=z[source],
        y_target=y[target],
        z_target=z[target],
        x_target=x[target],
        s=s,
    )


def simulate_cmc(
    n_total: int,
    key: Array,
    mechanism: str = "Mcor",
    membership: str = "Wcor",
) -> CMCSample:
    """One Section 5.1 replicate. X is latent on the target."""
    y, x, z, s = simulate_complete(n_total, key, mechanism, membership)
    return split_source_target(y, x, z, s)


def simulate_target_population(
    n_target: int,
    key: Array,
    mechanism: str = "Mcor",
    membership: str = "Wcor",
    max_draws: Optional[int] = None,
) -> tuple:
    """Complete (Y, X, Z) draws from the target population (S=0).

    Used as a large-sample benchmark for beta0, not as an estimator input
    with X observed in the analysis sample.
    """
    if max_draws is None:
        max_draws = n_target * 8
    y, x, z, s = simulate_complete(max_draws, key, mechanism, membership)
    target = ~s.astype(bool)
    if int(target.sum()) < n_target:
        raise RuntimeError(
            f"only {int(target.sum())} target rows in {max_draws} draws; increase max_draws"
        )
    return y[target][:n_target], x[target][:n_target], z[target][:n_target]
