"""Shared pytest fixtures."""

import pytest
from fastapi.testclient import TestClient

from raggate.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)