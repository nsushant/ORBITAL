"""SLA demonstrator components curated from the ORBITAL repository."""

__all__ = ["transfer_cost", "transfer_cost_grid", "transfer_dv"]


def __getattr__(name):
    if name in __all__:
        from . import edelbaum
        return getattr(edelbaum, name)
    raise AttributeError(name)
