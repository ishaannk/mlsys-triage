"""Jev evidence judge: wired into `retrieve` behind a switch, and the offline
risk swap in scripts/c1_jev.py agrees with the live allocator's arithmetic."""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from conftest import mk_items

from tri import actions as A
from tri import jev
from tri.actions import Ctx
from app.router.abstain import abstain_risk
from tri.substrate.mock import MockSubstrate

_spec = importlib.util.spec_from_file_location(
    "c1_jev", Path(__file__).resolve().parent.parent / "scripts" / "c1_jev.py")
c1_jev = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(c1_jev)


class StubJev:
    async def system_one(self, state, questions):
        assert "answer" in state and state["passages"]
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=100), model="stub",
            nouls={"sufficient": SimpleNamespace(noul=0.07)},
            choices={"relation": SimpleNamespace(
                choice="says_nothing", probabilities={"supports": 0.1, "contradicts": 0.2})})


def test_retrieve_uses_jev_only_when_switched_on(cfg, monkeypatch):
    monkeypatch.setattr(jev, "_client", StubJev())

    async def run(judge):
        c = {**cfg, "signals": {**cfg["signals"], "evidence_judge": judge}}
        sub = MockSubstrate.from_config({"tiers": cfg["tiers"], "mock": cfg["mock"]})
        await sub.start()
        ctx = Ctx(item=mk_items(1)[0], substrate=sub, cfg=c, seed=0)
        await A.apply(ctx, "answer_small")
        await A.apply(ctx, "retrieve")
        await sub.close()
        return ctx

    proxy, judged = asyncio.run(run("proxy")), asyncio.run(run("jev"))
    assert proxy.signals.evidence_sufficiency != 0.07
    assert judged.signals.evidence_sufficiency == 0.07
    d = judged.signals.detail["evidence_sufficiency"]
    assert d["proxy"] == proxy.signals.evidence_sufficiency and d["jev_tokens_in"] == 100

    # The offline swap, starting from the PROXY run's logged signals, must land
    # on the risk the live Jev run logs (abstain_risk, as policies.py records it).
    row = {"signals": proxy.signals.model_dump() if hasattr(proxy.signals, "model_dump")
           else vars(proxy.signals), "actions": proxy.actions_taken}
    a = cfg["abstain"]
    assert c1_jev.risk_with(row, a, proxy.signals.evidence_sufficiency) == abstain_risk(proxy.signals, a)[0]
    assert c1_jev.risk_with(row, a, 0.07) == abstain_risk(judged.signals, a)[0]
