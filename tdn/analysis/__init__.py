"""Audits and measured frontiers, with lightweight controller imports."""

__all__ = ["audit", "benchmark", "evaluate"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import workflow
    value = getattr(workflow, name)
    globals()[name] = value
    return value
