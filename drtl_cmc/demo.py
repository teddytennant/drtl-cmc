"""One correct-specification logistic replicate and one misspecified replicate.

Sample sizes are smaller than Section 5. Logistic conditional functionals are
Monte Carlo, not a closed form. No bootstrap standard errors.
"""

import jax
import jax.numpy as jnp

from drtl_cmc._jaxsetup import ensure_x64
from drtl_cmc.estimators import fit_dr, fit_glm, fit_imp, fit_iw, source_glm_start
from drtl_cmc.links import LOGISTIC
from drtl_cmc.nuisance import density_ratio_weights, fit_density_ratio, fit_gaussian_conditional
from drtl_cmc.simulate import simulate_cmc, simulate_target_population

ensure_x64()

N_TOTAL = 1200
K = 64
BENCHMARK_N = 20000


def _fmt(beta: jnp.ndarray) -> str:
    return " ".join(f"{float(v): .6f}" for v in beta)


def run_replicate(name: str, mechanism: str, membership: str, seed: int) -> None:
    key = jax.random.PRNGKey(seed)
    key_data, key_bench, key_est = jax.random.split(key, 3)
    sample = simulate_cmc(N_TOTAL, key_data, mechanism, membership)
    n = int(sample.y_source.shape[0])
    N = int(sample.y_target.shape[0])

    ratio = fit_density_ratio(
        sample.y_source, sample.z_source, sample.y_target, sample.z_target
    )
    omega = density_ratio_weights(ratio.eta, sample.y_source, sample.z_source)
    conditional = fit_gaussian_conditional(
        sample.y_source, sample.x_source, sample.z_source
    )
    start = source_glm_start(
        sample.y_source, sample.x_source, sample.z_source, LOGISTIC
    )
    _, k_imp, k_dr = jax.random.split(key_est, 3)
    iw = fit_iw(sample.y_source, sample.x_source, sample.z_source, omega, LOGISTIC, beta_init=start)
    imp = fit_imp(
        sample.y_target,
        sample.z_target,
        conditional,
        LOGISTIC,
        key=k_imp,
        K=K,
        beta_init=start,
    )
    dr = fit_dr(
        sample.y_source,
        sample.x_source,
        sample.z_source,
        sample.y_target,
        sample.z_target,
        omega,
        conditional,
        LOGISTIC,
        key=k_dr,
        K=K,
        beta_init=start,
    )

    y_b, x_b, z_b = simulate_target_population(
        BENCHMARK_N, key_bench, mechanism, membership
    )
    benchmark = fit_glm(y_b, x_b, z_b, LOGISTIC)

    print(
        f"{name} ({mechanism}, {membership}) seed={seed} n={n} N={N} K={K}"
    )
    print(f"  benchmark {_fmt(benchmark.beta)}")
    print(f"  DR        {_fmt(dr.beta)}  score_norm={float(dr.score_norm):.3e}")
    print(f"  IW        {_fmt(iw.beta)}  score_norm={float(iw.score_norm):.3e}")
    print(f"  IMP       {_fmt(imp.beta)}  score_norm={float(imp.score_norm):.3e}")
    print(f"  density-ratio FOC norm={float(ratio.foc_norm):.3e}")


def main() -> None:
    run_replicate("correct specification", "Mcor", "Wcor", seed=0)
    run_replicate("misspecified selection", "Mcor", "Wmis", seed=1)


if __name__ == "__main__":
    main()
