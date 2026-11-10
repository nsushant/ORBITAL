"""Public package surface."""

import mdls


def test_version_and_public_exports():
    assert mdls.__version__ == "0.1.0"
    for name in mdls.__all__:
        assert hasattr(mdls, name), name
