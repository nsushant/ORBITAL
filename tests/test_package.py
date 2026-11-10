"""Public package surface."""

import re
from importlib.metadata import PackageNotFoundError, version

import mdls


def test_version_and_public_exports():
    assert re.fullmatch(r"\d+\.\d+\.\d+", mdls.__version__), mdls.__version__
    try:
        installed = version("pymdls")
    except PackageNotFoundError:
        installed = None
    if installed is not None:
        assert mdls.__version__ == installed
    for name in mdls.__all__:
        assert hasattr(mdls, name), name
