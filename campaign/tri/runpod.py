"""Minimal RunPod client. Read-only by default; every spending call is explicit.

Two operational facts this wraps, both of which cost money to rediscover:

  * RunPod sits behind Cloudflare, which 403s the default Python user-agent with
    "error code: 1010". It is not an auth failure and looks exactly like one.
  * A NETWORK VOLUME IS PINNED TO ONE DATACENTER and cannot be moved. The plan
    (section 3.1) is explicit that this has killed campaigns: create the volume,
    then immediately verify BOTH card classes can launch in that DC, or destroy
    it (costs cents) and pick another.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional

import httpx

UA = "runpod-python/1.7 (triage-campaign)"
GQL = "https://api.runpod.io/graphql"
REST = "https://rest.runpod.io/v1"


def api_key() -> str:
    from dotenv import load_dotenv
    from . import ROOT
    load_dotenv(ROOT.parent / ".env")
    for k in ("runpod-api-key", "RUNPOD_API_KEY", "runpod_api_key"):
        v = os.getenv(k)
        if v:
            return v.strip()
    raise RuntimeError("no RunPod API key in .env (expected `runpod-api-key`)")


@dataclass
class RunPod:
    key: str = ""
    timeout: float = 60.0

    def __post_init__(self) -> None:
        self.key = self.key or api_key()

    @property
    def _h(self) -> dict[str, str]:
        return {"Authorization": "Bearer " + self.key, "User-Agent": UA,
                "Content-Type": "application/json"}

    # -- raw ------------------------------------------------------------- #
    def gql(self, query: str, variables: Optional[dict] = None) -> dict[str, Any]:
        r = httpx.post(GQL, headers=self._h, timeout=self.timeout,
                       json={"query": query, "variables": variables or {}})
        r.raise_for_status()
        d = r.json()
        if d.get("errors"):
            raise RuntimeError("graphql: %s" % json.dumps(d["errors"])[:400])
        return d.get("data") or {}

    def rest(self, method: str, path: str, **kw) -> Any:
        r = httpx.request(method, REST + path, headers=self._h,
                          timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise RuntimeError("REST %s %s -> %d: %s"
                               % (method, path, r.status_code, r.text[:400]))
        return r.json() if r.content else None

    # -- read-only -------------------------------------------------------- #

    def gpu_types(self) -> list[dict[str, Any]]:
        q = """{ gpuTypes { id displayName memoryInGb secureCloud communityCloud
                 lowestPrice(input:{gpuCount:1}) { minimumBidPrice uninterruptablePrice } } }"""
        return self.gql(q)["gpuTypes"]

    def pods(self) -> list[dict[str, Any]]:
        return self.rest("GET", "/pods") or []

    def volumes(self) -> list[dict[str, Any]]:
        return self.rest("GET", "/networkvolumes") or []

    # -- spending: every one of these costs money --------------------------- #
    def create_volume(self, name: str, size_gb: int, datacenter_id: str) -> dict[str, Any]:
        return self.rest("POST", "/networkvolumes",
                         json={"name": name, "size": int(size_gb),
                               "dataCenterId": datacenter_id})

    def delete_volume(self, volume_id: str) -> Any:
        return self.rest("DELETE", "/networkvolumes/" + volume_id)

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        return self.rest("POST", "/pods", json=body)

    def pod(self, pod_id: str) -> dict[str, Any]:
        return self.rest("GET", "/pods/" + pod_id)

    def stop_pod(self, pod_id: str) -> Any:
        return self.rest("POST", "/pods/%s/stop" % pod_id)

    def delete_pod(self, pod_id: str) -> Any:
        return self.rest("DELETE", "/pods/" + pod_id)
