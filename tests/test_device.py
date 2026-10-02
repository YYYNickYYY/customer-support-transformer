from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from utils import device


@pytest.mark.parametrize(
    "preference,available,expected",
    [("auto", True, "cuda"), ("auto", False, "cpu"),
     ("cuda", True, "cuda"), (" CUDA ", True, "cuda")],
)
def test_device_selection(monkeypatch, preference, available, expected):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))
    monkeypatch.setattr(device, "import_module", lambda name: torch)
    assert device.get_device(preference) == expected


def test_cpu_does_not_import_torch(monkeypatch):
    loader = Mock(side_effect=AssertionError("must not import"))
    monkeypatch.setattr(device, "import_module", loader)
    assert device.get_device("cpu") == "cpu"
    loader.assert_not_called()


def test_missing_optional_torch(monkeypatch):
    monkeypatch.setattr(device, "import_module", Mock(
        side_effect=ModuleNotFoundError("No module named torch", name="torch")
    ))
    assert device.get_device() == "cpu"
    with pytest.raises(RuntimeError, match="not installed"):
        device.get_device("cuda")


def test_explicit_unavailable_cuda(monkeypatch):
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(device, "import_module", lambda name: torch)
    with pytest.raises(RuntimeError, match="unavailable"):
        device.get_device("cuda")


@pytest.mark.parametrize("error", [
    ModuleNotFoundError("missing internal dependency", name="torch_dependency"),
    OSError("broken binary"),
])
def test_broken_installation_is_not_hidden(monkeypatch, error):
    monkeypatch.setattr(device, "import_module", Mock(side_effect=error))
    with pytest.raises(type(error)) as raised:
        device.get_device()
    assert raised.value is error


def test_probe_error_propagates(monkeypatch):
    probe = Mock(side_effect=RuntimeError("probe failed"))
    monkeypatch.setattr(device, "import_module", lambda name:
                        SimpleNamespace(cuda=SimpleNamespace(is_available=probe)))
    with pytest.raises(RuntimeError, match="probe failed"):
        device.get_device()


@pytest.mark.parametrize("preference", ["", "mps", "cuda:0"])
def test_invalid_preference(preference):
    with pytest.raises(ValueError):
        device.get_device(preference)


def test_non_string_preference():
    with pytest.raises(TypeError):
        device.get_device(None)
