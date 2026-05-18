import logging
import os
import re
from collections.abc import Awaitable, Callable
from pathlib import Path

from bluebird_dt import logger
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles

from .route_tags import tags_metadata
from .routes import router

bluebird_logger = logging.getLogger("bluebird_dt")
bluebird_logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logger.CustomFormatter())
bluebird_logger.addHandler(stream_handler)

app = FastAPI(
    title="BluebirdATC: AI for air traffic control",
    description="FastAPI interface to control the simulation framework BluebirdATC.",
    license_info={
        "name": "License",
        "url": "https://github.com/project-bluebird/BluebirdATC/blob/dev/LICENSE",
    },
    openapi_tags=tags_metadata,
    strict_content_type=False,
)
app.include_router(router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)

# serve the frontend
package_root = Path(__file__).resolve().parents[1]
dist_path = package_root / "bluebird_api" / "hmi"

if dist_path.is_dir():
    app.mount("/hmi", StaticFiles(directory=str(dist_path), html=True), name="frontend")
    app.mount("/hmi/assets", StaticFiles(directory=os.path.join(dist_path, "assets")), name="assets")


# For webapp, requests will be prepended with "/api" - strip that out here
@app.middleware("http")
async def strip_api_prefix(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    path = request.url.path
    if path.startswith("/api"):
        newpath = re.sub("/api", "", path)
        request.scope["path"] = newpath
    return await call_next(request)


def use_route_names_as_operation_ids(app: FastAPI) -> None:
    """
    Simplify operation IDs so that generated API clients have simpler function
    names.

    Must be called after all routes have been added.
    """
    for route in app.routes:
        if isinstance(route, APIRoute):
            route.operation_id = route.name


use_route_names_as_operation_ids(app)

# keep track of current simulation instance
# app.state.current_runner = None
