"""Tests for backend_factory.get_backend()."""
import pytest

from backend_factory import get_backend
from vllm_backend import VllmBackend


def test_default_backend_is_vllm(monkeypatch):
    monkeypatch.delenv("GUARD_BACKEND", raising=False)
    assert isinstance(get_backend(), VllmBackend)


def test_explicit_vllm(monkeypatch):
    monkeypatch.setenv("GUARD_BACKEND", "vllm")
    assert isinstance(get_backend(), VllmBackend)


def test_vllm_case_and_whitespace_insensitive(monkeypatch):
    monkeypatch.setenv("GUARD_BACKEND", "  VLLM  ")
    assert isinstance(get_backend(), VllmBackend)


def test_realtime_raises_not_implemented(monkeypatch):
    monkeypatch.setenv("GUARD_BACKEND", "realtime")
    with pytest.raises(NotImplementedError, match="OpenBMB realtime server"):
        get_backend()


def test_unknown_backend_raises_value_error(monkeypatch):
    monkeypatch.setenv("GUARD_BACKEND", "not-a-backend")
    with pytest.raises(ValueError, match="not-a-backend"):
        get_backend()
