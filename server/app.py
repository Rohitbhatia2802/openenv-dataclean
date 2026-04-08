"""
FastAPI Server for the OpenEnv Data Cleaning Environment.

Exposes the four required OpenEnv HTTP endpoints:
  POST /reset   — start a new episode
  POST /step    — execute one action
  GET  /state   — inspect full internal state
  GET  /health  — service health check

Thread safety note: Each HTTP request shares the same DataCleaningEnv
instance. For concurrent load-testing, wrap with asyncio.Lock or run
multiple worker processes behind a load balancer.

Usage::

    uvicorn server:app --host 0.0.0.0 --port 7860 --workers 1
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.env import DataCleaningEnv, ALLOWED_TASK_IDS
from src.models import Observation, Reward, State

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("openenv.server")

# ---------------------------------------------------------------------------
# Request / Response schemas for FastAPI
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    """Body for POST /reset."""
    task_id: str = Field(
        ...,
        description=f"One of: {ALLOWED_TASK_IDS}",
        examples=["fix_missing_price"],
    )


class StepRequest(BaseModel):
    """Body for POST /step."""
    operation: str = Field(..., description="Operation type string")
    target_columns: list[str] = Field(
        default_factory=list, description="Columns to apply operation to"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Operation-specific parameters"
    )


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str
    version: str
    active_task: str | None
    uptime_seconds: float
    allowed_task_ids: list[str]


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

_START_TIME = time.time()
_env = DataCleaningEnv(seed=42, verbose=False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("OpenEnv Data Cleaning server starting up...")
    yield
    logger.info("OpenEnv Data Cleaning server shutting down.")


app = FastAPI(
    title="OpenEnv — Data Cleaning & Validation",
    description=(
        "Production-grade OpenEnv environment simulating real-world enterprise "
        "data cleaning pipelines. Three difficulty tiers: easy, medium, hard."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Exception handler — return JSON instead of HTML on unexpected errors
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "path": str(request.url.path)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Infrastructure"])
async def health() -> HealthResponse:
    """
    Service health check.

    Returns server status, uptime, and the list of registered task IDs.
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        active_task=_env._task_id,
        uptime_seconds=round(time.time() - _START_TIME, 2),
        allowed_task_ids=ALLOWED_TASK_IDS,
    )


@app.post("/reset", response_model=Observation, tags=["OpenEnv API"])
async def reset(body: ResetRequest | None = None) -> Observation:
    """
    Reset the environment and begin a new episode.

    - **task_id**: Identifier of the task to run. (Defaults to 'fix_missing_price' if body is missing)

    Returns the initial observation of the fresh dataset.
    """
    task_id = body.task_id if body else "fix_missing_price"
    try:
        obs = _env.reset(task_id)
        logger.info("Episode reset: task_id=%s", task_id)
        return obs
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/step", tags=["OpenEnv API"])
async def step(body: StepRequest) -> Dict[str, Any]:
    """
    Execute one action in the environment.

    Returns:
    - **observation**: Updated environment observation
    - **reward**: Step reward with breakdown
    - **done**: Whether the episode has ended
    - **info**: Auxiliary diagnostic information
    """
    if _env._task_id is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )

    action_dict = {
        "operation": body.operation,
        "target_columns": body.target_columns,
        "params": body.params,
    }

    obs, reward, done, info = _env.step(action_dict)

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state", response_model=State, tags=["OpenEnv API"])
async def state() -> State:
    """
    Retrieve the full internal state snapshot.

    Returns the complete episode state including history, DataFrame metadata,
    defect counts, and the current observation. Useful for debugging and
    human review.
    """
    if _env._task_id is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    """Run the FastAPI server."""
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=port,
        workers=1,
        log_level="info",
        reload=False,
    )


if __name__ == "__main__":
    main()
