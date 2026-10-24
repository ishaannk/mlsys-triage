"""VLLMSubstrate against a stub server. Proves, on CPU:

  * TTFT comes from the first STREAMED chunk and is strictly less than e2e
    (the bug the whole substrate exists to fix);
  * the equal-split GPU-second accrual conserves exactly under concurrency;
  * a dead engine produces an error, never a plausible-looking row;
  * the substrate REFUSES to start when a tier is unreachable, instead of
    silently degrading to mock.
"""
from __future__ import annotations

import asyncio

import pytest

from stub_vllm import StubServer

from tri.substrate.vllm import SubstrateUnavailable, VLLMSubstrate
from tri.types import GenRequest, TierSpec


def _sub(base: str, served: str = "small") -> VLLMSubstrate:
    return VLLMSubstrate({served: TierSpec(name=served, model_id="stub/model",
                                           kind="local_gpu", endpoint=base,
                                           served_model_name=served)},
                         metrics_hz=20.0)


def test_ttft_comes_from_the_first_streamed_chunk():
    async def run():
        with StubServer() as srv:
            sub = _sub(srv.base)
            await sub.start()
            out = await sub.generate(GenRequest(
                prompt=[{"role": "user", "content": "hello"}], tier="small",
                want_logprobs=True, max_tokens=20))
            a = out.accounting
            assert out.error is None, out.error
            assert out.text.startswith("Answer: B")
            assert a.output_tokens == 20
            assert a.prompt_tokens == 123, "usage must come from the server"
            assert a.cached_prompt_tokens == 40, "prefix-cache hits must be recorded"
            assert out.logprobs and len(out.logprobs) == 20
            # the whole point:
            assert a.ttft_ms < a.e2e_ms
            assert a.ttft_ms >= 40, "TTFT should reflect the 50 ms stub prefill"
            assert a.e2e_ms - a.ttft_ms >= 50, "decode time must be visible"
            await sub.close()
    asyncio.run(run())


def test_equal_split_accrual_conserves_under_concurrency():
    async def run():
        with StubServer() as srv:
            sub = _sub(srv.base)
            await sub.start()
            outs = await asyncio.gather(*(
                sub.generate(GenRequest(prompt=[{"role": "user", "content": "q%d" % i}],
                                        tier="small", max_tokens=12))
                for i in range(8)))
            attributed = sum(o.accounting.gpu_seconds_share for o in outs)
            assert sub.busy_s > 0
            rel = abs(attributed - sub.busy_s) / sub.busy_s
            assert rel < 1e-9, "equal-split accrual does not conserve: %g" % rel
            # concurrency means each request is charged LESS than its wall time
            for o in outs:
                a = o.accounting
                assert a.gpu_seconds_share < a.e2e_ms / 1000.0 * 1.01
            await sub.close()
    asyncio.run(run())


def test_engine_failure_is_an_error_not_a_plausible_row():
    async def run():
        with StubServer() as srv:
            sub = _sub(srv.base)
            await sub.start()
            srv.handler.fail_next = True
            out = await sub.generate(GenRequest(
                prompt=[{"role": "user", "content": "x"}], tier="small"))
            assert out.error and "500" in out.error
            assert out.text == ""
            assert out.accounting.finish_reason == "error"
            assert out.accounting.admitted is False
            await sub.close()
    asyncio.run(run())


def test_refuses_to_start_when_a_tier_is_unreachable():
    async def run():
        sub = _sub("http://127.0.0.1:9/v1")        # discard port: nothing listens
        with pytest.raises(SubstrateUnavailable, match="REFUSING to fall back to mock"):
            await sub.start()
        await sub.close()
    asyncio.run(run())


def test_refuses_when_the_served_model_name_is_wrong():
    """A tier serving the wrong model silently makes every 'small vs big'
    comparison a comparison of one model with itself."""
    async def run():
        with StubServer(served_model="something_else") as srv:
            sub = _sub(srv.base, served="small")
            with pytest.raises(SubstrateUnavailable, match="expected"):
                await sub.start()
            await sub.close()
    asyncio.run(run())


def test_state_reads_live_metrics_and_tracks_staleness():
    async def run():
        with StubServer() as srv:
            sub = _sub(srv.base)
            await sub.start()
            await asyncio.sleep(0.3)               # let the scraper land one scrape
            st = sub.state()
            ts = st.per_tier["small"]
            assert ts.num_running == 7 and ts.num_waiting == 3
            assert abs(ts.kv_usage - 0.6423) < 1e-6
            assert st.staleness_ms < 1000
            assert ts.healthy
            await sub.close()
    asyncio.run(run())


def test_only_serving_tiers_are_selected_and_health_checked():
    """A tier that is not running must not be `local_gpu`.

    from_config selects on kind and start() then health-checks everything it
    selected, refusing to run if any is unreachable. That is right - but it means
    a staged campaign that brings tiers up one phase at a time has to say which
    are live. Leaving all three set to local_gpu while only `mid` served made the
    substrate refuse to start against two tiers the phase never calls.
    """
    from tri.config import load
    from tri.substrate.vllm import VLLMSubstrate
    cfg = load()
    for t in cfg["tiers"].values():
        t["kind"] = "offline"
    cfg["tiers"]["mid"]["kind"] = "local_gpu"
    assert sorted(VLLMSubstrate.from_config(cfg).tiers) == ["mid"]

    cfg["tiers"]["small"]["kind"] = cfg["tiers"]["big"]["kind"] = "local_gpu"
    cfg["tiers"]["mid"]["kind"] = "offline"
    assert sorted(VLLMSubstrate.from_config(cfg).tiers) == ["big", "small"]
