from pathlib import Path

import pytest

from modal_app import _volume_path


def test_volume_path_resolves_relative_paths():
    assert _volume_path(
        Path("/mnt/data"), "features/dino336-v1/train"
    ) == Path("/mnt/data/features/dino336-v1/train")


@pytest.mark.parametrize(
    "value",
    ("", ".", "../secrets", "features/../../secrets", "/etc/passwd"),
)
def test_volume_path_rejects_unsafe_paths(value):
    with pytest.raises(ValueError, match="relative paths"):
        _volume_path(Path("/mnt/data"), value)
