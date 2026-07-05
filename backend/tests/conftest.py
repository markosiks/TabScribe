import pytest
from fastapi import FastAPI

from backend.app.main import app


@pytest.fixture()
def asgi_app() -> FastAPI:
    return app
