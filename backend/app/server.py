from __future__ import annotations

import uvicorn

from app.core.config import settings


def run() -> None:
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
