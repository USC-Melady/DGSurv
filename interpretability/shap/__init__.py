
# explainers
from .explainers._deep import DeepExplainer
try:
    # Version from setuptools-scm
    from ._version import version as __version__
except ImportError:
    # Expected when running locally without build
    __version__ = "0.0.0-not-built"

_no_matplotlib_warning = "matplotlib is not installed so plotting is not available! Run `pip install matplotlib` " \
                         "to fix this."


# plotting (only loaded if matplotlib is present)
def unsupported(*args, **kwargs):
    raise ImportError(_no_matplotlib_warning)


class UnsupportedModule:
    def __getattribute__(self, item):
        raise ImportError(_no_matplotlib_warning)



# Use __all__ to let type checkers know what is part of the public API.
__all__ = [
    "DeepExplainer",
]
