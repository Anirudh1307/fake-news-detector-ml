"""Application package for Fake News Detector."""

import logging
import os

from app.routes import create_app

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

app = create_app()

__all__ = ["app", "create_app"]
