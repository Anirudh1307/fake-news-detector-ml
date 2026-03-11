"""Application package for Fake News Detector."""

from app.routes import create_app

app = create_app()

__all__ = ["app", "create_app"]
