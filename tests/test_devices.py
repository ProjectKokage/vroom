from __future__ import annotations

import pytest

from vroom.errors import DeviceError
from vroom.hy_motion import select_device


class _Cuda:
    def __init__(self, available: bool, count: int = 0):
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _Mps:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _Backends:
    def __init__(self, mps: bool):
        self.mps = _Mps(mps)


class _Torch:
    def __init__(self, *, cuda: bool = False, count: int = 0, mps: bool = False):
        self.cuda = _Cuda(cuda, count)
        self.backends = _Backends(mps)


def test_auto_device_priority() -> None:
    assert select_device("auto", _Torch(cuda=True, count=2, mps=True)) == "cuda:0"
    assert select_device("auto", _Torch(mps=True)) == "mps"
    assert select_device("auto", _Torch()) == "cpu"


def test_explicit_cuda_index() -> None:
    assert select_device("cuda:1", _Torch(cuda=True, count=2)) == "cuda:1"
    with pytest.raises(DeviceError, match="outside"):
        select_device("cuda:2", _Torch(cuda=True, count=2))


def test_unavailable_explicit_accelerator_fails() -> None:
    with pytest.raises(DeviceError, match="MPS"):
        select_device("mps", _Torch())
    with pytest.raises(DeviceError, match="CUDA"):
        select_device("cuda", _Torch())
