"""GLM mean functions. g is the inverse link: E[Y | X, Z] = g(X' beta_x + Z' beta_z)."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from drtl_cmc._jaxsetup import ensure_x64

ensure_x64()

Array = jnp.ndarray


class Link(NamedTuple):
    name: str
    # 0 identity, 1 logistic. Passed as a static JIT argument.
    link_id: int
    g: Callable[[Array], Array]
    g_deriv: Callable[[Array], Array]
    closed_form_gaussian: bool


def _identity(u: Array) -> Array:
    return u


def _identity_deriv(u: Array) -> Array:
    return jnp.ones_like(u)


def _logistic(u: Array) -> Array:
    return jax.nn.sigmoid(u)


def _logistic_deriv(u: Array) -> Array:
    p = jax.nn.sigmoid(u)
    return p * (1.0 - p)


IDENTITY = Link("identity", 0, _identity, _identity_deriv, True)
LOGISTIC = Link("logistic", 1, _logistic, _logistic_deriv, False)


def get_link(name: str) -> Link:
    key = name.lower()
    if key in ("identity", "linear"):
        return IDENTITY
    if key in ("logistic", "logit", "sigmoid"):
        return LOGISTIC
    raise ValueError(f"unknown link {name!r}; expected 'identity' or 'logistic'")
