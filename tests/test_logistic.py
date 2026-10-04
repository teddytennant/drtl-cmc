"""Section 5.1 logistic design, smaller than the paper.

Correct specification (Mcor, Wcor): DR, IW, and IMP with estimated nuisances
land near a complete-data target logistic MLE. One misspecified configuration
(Mcor, Wmis): IW, which uses the wrong density ratio, has larger error than DR
on beta_z3, the coefficient the paper shows is sensitive to that misspecification.
"""

import jax
import jax.numpy as jnp
import numpy as np

from drtl_cmc.estimators import fit_dr, fit_glm, fit_imp, fit_iw, source_glm_start
from drtl_cmc.links import LOGISTIC
from drtl_cmc.nuisance import (
    density_ratio_weights,
    fit_density_ratio,
    fit_gaussian_conditional,
)
from drtl_cmc.scores import complete_score_and_jac
from drtl_cmc.simulate import simulate_cmc, simulate_target_population

K = 128
N_TOTAL = 1500


def _numpy_logistic_score(beta, y, x, z):
    """Independent check of the logistic complete-data score. Not the library."""
    u = np.concatenate([np.asarray(x).reshape(-1, 1), np.asarray(z)], axis=1)
    eta = u @ np.asarray(beta)
    prob = 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
    return (u * (np.asarray(y) - prob)[:, None]).mean(axis=0)


def _benchmark(mechanism, membership, seed, n_target=40_000):
    y, x, z = simulate_target_population(
        n_target, jax.random.PRNGKey(seed), mechanism, membership
    )
    fit = fit_glm(y, x, z, LOGISTIC)
    score = _numpy_logistic_score(fit.beta, y, x, z)
    assert np.linalg.norm(score) < 1e-8
    assert float(fit.score_norm) < 1e-8
    return fit.beta


def _fit_all(sample, key, K=K):
    ratio = fit_density_ratio(
        sample.y_source, sample.z_source, sample.y_target, sample.z_target
    )
    assert float(ratio.foc_norm) < 1e-8
    omega = density_ratio_weights(ratio.eta, sample.y_source, sample.z_source)
    conditional = fit_gaussian_conditional(
        sample.y_source, sample.x_source, sample.z_source
    )
    start = source_glm_start(
        sample.y_source, sample.x_source, sample.z_source, LOGISTIC
    )
    k_imp, k_dr = jax.random.split(key)
    iw = fit_iw(
        sample.y_source,
        sample.x_source,
        sample.z_source,
        omega,
        LOGISTIC,
        beta_init=start,
    )
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
    return dr, iw, imp


def test_logistic_imputed_jacobian_matches_finite_difference():
    """Analytic Jacobian of the Monte Carlo imputed score, fixed draws."""
    from drtl_cmc.nuisance import GaussianConditional
    from drtl_cmc.scores import mc_imputed_score_and_jac

    n, p, q, K = 20, 1, 3, 32
    conditional = GaussianConditional(
        B=jnp.array([[0.4], [0.2], [-0.3], [0.1]]),
        Sigma=jnp.array([[0.25]]),
    )
    y = jax.random.bernoulli(jax.random.PRNGKey(1), 0.5, (n,)).astype(jnp.float64)
    z = jnp.concatenate(
        [jnp.ones((n, 1)), jax.random.normal(jax.random.PRNGKey(2), (n, q - 1))],
        axis=1,
    )
    beta = jnp.array([0.8, -0.3, 0.4, -0.2])
    draws = jax.random.normal(jax.random.PRNGKey(3), (K, n, p))
    _, jac = mc_imputed_score_and_jac(
        y, z, beta, conditional, draws, LOGISTIC.link_id
    )
    step = 1e-6

    def mean_score(b):
        return mc_imputed_score_and_jac(
            y, z, b, conditional, draws, LOGISTIC.link_id
        )[0].mean(0)

    cols = []
    for j in range(beta.size):
        cols.append((mean_score(beta.at[j].add(step)) - mean_score(beta)) / step)
    assert jnp.allclose(jac.mean(0), jnp.stack(cols, 1), atol=1e-5)


def test_logistic_jacobian_matches_finite_difference():
    key = jax.random.PRNGKey(4)
    n = 30
    x = jax.random.normal(key, (n, 1))
    z = jnp.concatenate(
        [jnp.ones((n, 1)), jax.random.normal(jax.random.PRNGKey(5), (n, 2))],
        axis=1,
    )
    y = jax.random.bernoulli(jax.random.PRNGKey(6), 0.4, (n,)).astype(jnp.float64)
    beta = jnp.array([0.4, -0.2, 0.3, -0.1])
    phi, jac = complete_score_and_jac(y, x, z, beta, LOGISTIC.link_id)
    eps = 1e-6

    def mean_phi(b):
        return complete_score_and_jac(y, x, z, b, LOGISTIC.link_id)[0].mean(0)

    cols = []
    for j in range(beta.size):
        cols.append((mean_phi(beta.at[j].add(eps)) - mean_phi(beta)) / eps)
    assert jnp.allclose(jac.mean(0), jnp.stack(cols, 1), atol=1e-5)
    assert phi.shape == (n, 4)


def test_correct_specification_near_target_mle():
    beta0 = _benchmark("Mcor", "Wcor", seed=50)
    # Zeros are far from every large coefficient of this design.
    assert float(jnp.max(jnp.abs(beta0))) > 1.0

    sample = simulate_cmc(N_TOTAL, jax.random.PRNGKey(42), "Mcor", "Wcor")
    assert 800 <= sample.y_source.shape[0] + sample.y_target.shape[0] <= 1500
    dr, iw, imp = _fit_all(sample, jax.random.PRNGKey(6))
    for name, fit in (("DR", dr), ("IW", iw), ("IMP", imp)):
        assert float(fit.score_norm) < 1e-6, name
        err = jnp.abs(fit.beta - beta0)
        # A few tenths. A zero vector fails: |beta_x| and |beta_z2| exceed 1.
        assert float(jnp.max(err)) < 0.40, (name, fit.beta, beta0, err)
        assert float(jnp.linalg.norm(fit.beta)) > 1.0


def test_wmis_iw_worse_than_dr_on_beta_z3():
    """Configuration (III): q correct, w misspecified. Paper: IW bias on beta_z3.

    A handful of seeds, still small n. The check is the average absolute error,
    not a single noisy draw.
    """
    beta0 = _benchmark("Mcor", "Wmis", seed=70)
    z3 = 3  # beta = (beta_x, beta_z1, beta_z2, beta_z3)
    seeds = (11, 12, 13, 14, 15, 16)
    iw_err = []
    dr_err = []
    imp_err = []
    for seed in seeds:
        sample = simulate_cmc(N_TOTAL, jax.random.PRNGKey(seed), "Mcor", "Wmis")
        dr, iw, imp = _fit_all(sample, jax.random.PRNGKey(1000 + seed), K=64)
        assert float(dr.score_norm) < 1e-5
        assert float(iw.score_norm) < 1e-5
        iw_err.append(float(iw.beta[z3] - beta0[z3]))
        dr_err.append(float(dr.beta[z3] - beta0[z3]))
        imp_err.append(float(imp.beta[z3] - beta0[z3]))
    iw_err = np.asarray(iw_err)
    dr_err = np.asarray(dr_err)
    imp_err = np.asarray(imp_err)
    # Systematic IW bias should dominate DR. Mean absolute error, not |mean|,
    # so a sign flip across seeds cannot hide a large IW error.
    assert float(np.mean(np.abs(iw_err))) > float(np.mean(np.abs(dr_err))) + 0.05
    # The paper's sensitive direction is negative IW bias on beta_z3.
    assert float(np.mean(iw_err)) < -0.1
    assert float(np.mean(np.abs(imp_err))) < float(np.mean(np.abs(iw_err)))
