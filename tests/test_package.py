import tsfm


def test_version_is_exposed():
    """The installed package metadata is reachable, so the build backend packaged `src/tsfm`."""
    assert tsfm.__version__ == "0.1.0"
