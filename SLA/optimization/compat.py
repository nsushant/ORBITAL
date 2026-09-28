"""Optional numba acceleration with a pure-Python fallback."""
try:
    from numba import njit  # type: ignore
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda function: function
