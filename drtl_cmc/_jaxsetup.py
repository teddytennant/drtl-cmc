"""Enable float64 before any arrays are created."""

import jax

_ENABLED = False


def ensure_x64() -> None:
    global _ENABLED
    if not _ENABLED:
        jax.config.update("jax_enable_x64", True)
        _ENABLED = True


ensure_x64()
