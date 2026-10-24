"""Triage FastAPI backend."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .benchmark.harness import run_benchmark
from .config import get_settings, load_models, load_router_config
from .llm import RateLimitError
from .providers.openai_compat import USER_OPENAI_KEY
from .providers.registry import get_registry
from .retrieval.store import get_store, store_info
from .router import memory, online
from .router.router import route_and_answer
from .schemas import BenchmarkRequest, ChatRequest, ChatResponse, IngestRequest
from .telemetry import anomaly, db

ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"

app = FastAPI(title="Triage", version="1.0", description="Reliability-aware compute router for LLM serving")


@app.on_event("startup")
def _startup() -> None:
    db.init_db()
    memory.init_db()          # semantic routing memory table + cache
    get_store()  # seed corpus
    reg = get_registry()
    print(f"[Triage] provider keys: {reg.status()}")
    print(f"[Triage] retrieval: {store_info()}")
    print(f"[Triage] routing memory: {memory.size()} entries")


@app.get("/health")
def health() -> dict:
    reg = get_registry()
    return {
        "status": "ok",
        "providers": reg.status(),
        "force_mock": get_settings().force_mock,
        "retrieval": store_info(),
    }


@app.get("/models")
def models() -> dict:
    reg = get_registry()
    out = []
    for m in load_models():
        adapter, provider_model, label = reg.resolve(m["id"])
        out.append({**m, "live_provider": label, "live_model": provider_model,
                    "is_mock": label == "mock"})
    return {"models": out, "provider_status": reg.status()}


@app.post("/chat", response_model=None)
async def chat(req: ChatRequest):
    if req.api_key:                       # BYO key: request-scoped, never persisted
        USER_OPENAI_KEY.set(req.api_key)
        req.api_key = None                # keep it out of telemetry/logs
    try:
        resp, telem = await route_and_answer(req)
    except RateLimitError as e:
        return JSONResponse(status_code=429, content={
            "error": "rate_limited", "provider": e.provider,
            "retry_after_seconds": e.retry_after,
            "message": ("The free model lane just hit the provider's rate limit "
                        "(NVIDIA free tier allows ~40 requests/min). It resets in "
                        "about a minute — please try again shortly."),
        })
    db.log_request(telem)
    return resp


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    added = get_store().add(req.documents, source=req.source)
    return {"added": added, "store": store_info()}


@app.post("/benchmark")
async def benchmark(req: BenchmarkRequest) -> JSONResponse:
    result = await run_benchmark(req.model, req.limit, req.big_model)
    return JSONResponse(result)


@app.get("/telemetry")
def telemetry(limit: int = 50) -> dict:
    return {"summary": db.summary(), "recent": db.recent(limit)}


@app.get("/anomalies")
def anomalies() -> dict:
    """Cost Anomaly Detection over the telemetry stream."""
    cfg = load_router_config().get("anomaly", {}) or {}
    return anomaly.detect(window=cfg.get("window", 200), z_threshold=cfg.get("z_threshold", 3.0))


@app.get("/learning")
def learning() -> dict:
    """Online Learning Router state: the per-model escalation threshold it has learned,
    plus the Semantic Routing Memory size."""
    return {"online_learning": online.snapshot(), "routing_memory_entries": memory.size()}


# --- Static UI ------------------------------------------------------------- #
# Landing page = the demo/story page; the chat app lives at /app.
@app.get("/")
def landing() -> FileResponse:
    return FileResponse(UI_DIR / "demo.html")


@app.get("/app")
def chat_app() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")
