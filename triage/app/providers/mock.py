"""Mock provider — the offline fallback so Triage runs with zero API keys.

It returns deterministic answers from a tiny built-in knowledge base, plus
*synthetic* per-token logprobs, and it deliberately injects variability at
higher temperature for questions it is unsure about. That makes every proxy
signal (uncertainty, instability, contradiction, retrieval-disagreement,
evidence) exercise real code paths without any network call.
"""
from __future__ import annotations

import hashlib
import math
import random
import time

from .base import GenResult, Message, ProviderAdapter

# Tiny KB: keyword -> (answer, confidence 0..1). High confidence => stable + tight
# logprobs; low confidence => the model wavers across resamples.
_KB: list[tuple[tuple[str, ...], str, float]] = [
    (("capital", "france"), "The capital of France is Paris.", 0.98),
    (("capital", "japan"), "The capital of Japan is Tokyo.", 0.98),
    (("capital", "australia"), "The capital of Australia is Canberra.", 0.72),
    (("boiling", "water"), "Water boils at 100 degrees Celsius at sea level.", 0.95),
    (("speed", "light"), "The speed of light is about 299,792 km/s.", 0.95),
    (("author", "hamlet"), "Hamlet was written by William Shakespeare.", 0.95),
    (("largest", "planet"), "Jupiter is the largest planet in the Solar System.", 0.95),
    (("square", "root", "144"), "The square root of 144 is 12.", 0.97),
    (("2", "2"), "2 + 2 = 4.", 0.99),
    (("who", "president"), "That depends on the year; I am not certain of the current officeholder.", 0.35),
    (("cure", "cancer"), "There is no single cure for cancer; treatment depends on the type and stage.", 0.5),
    (("meaning", "life"), "There is no consensus answer; interpretations vary widely.", 0.3),
]
_HEDGE = ("I'm not fully certain, but ", "Possibly ", "It may be that ", "One common answer is ")


class MockProvider(ProviderAdapter):
    name = "mock"

    def available(self) -> bool:
        return True

    async def generate(
        self,
        model: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 512,
        want_logprobs: bool = False,
    ) -> GenResult:
        t0 = time.perf_counter()
        prompt = " ".join(m["content"] for m in messages if m["role"] == "user").lower()

        answer, conf = self._lookup(prompt)

        # Self-check probes ask the model to critique a prior answer; emulate that.
        if "does the following answer contain" in prompt or "self-check" in prompt:
            answer = "No contradiction found." if conf > 0.6 else "The answer may be inconsistent."
            conf = max(conf, 0.6)

        # Inject instability for low-confidence answers at nonzero temperature.
        rng = random.Random(f"{prompt}|{temperature}|{time.time_ns()}")
        if conf < 0.75 and temperature > 0.25 and rng.random() > conf:
            answer = rng.choice(_HEDGE) + answer

        tokens_out = max(1, len(answer.split()))
        tokens_in = max(1, len(prompt.split()))
        logprobs = self._synth_logprobs(answer, conf, want_logprobs)

        # Simulate a little latency proportional to output size.
        latency = 20 + tokens_out * 1.5 + rng.random() * 15
        return GenResult(
            text=answer, provider=self.name, model=model,
            tokens_in=tokens_in, tokens_out=tokens_out,
            ttft_ms=latency * 0.4, latency_ms=(time.perf_counter() - t0) * 1000 + latency,
            logprobs=logprobs, raw={"mock": True, "confidence": conf},
        )

    @staticmethod
    def _lookup(prompt: str) -> tuple[str, float]:
        best: tuple[str, float] | None = None
        best_hits = 0
        for keys, ans, conf in _KB:
            hits = sum(1 for k in keys if k in prompt)
            if hits > best_hits and hits >= max(1, len(keys) - 1):
                best, best_hits = (ans, conf), hits
        if best:
            return best
        # Unknown question: deterministic pseudo-answer with mid/low confidence.
        h = hashlib.sha256(prompt.encode()).hexdigest()
        conf = 0.35 + (int(h[:2], 16) / 255) * 0.35
        return (f"Based on general knowledge, a reasonable answer is: {prompt[:60].strip()}…", conf)

    @staticmethod
    def _synth_logprobs(answer: str, conf: float, want: bool) -> list[float] | None:
        if not want:
            return None
        # Higher confidence => logprobs closer to 0 (prob near 1); lower => more negative.
        base = -(1.0 - conf) * 1.8
        out = []
        for i, _tok in enumerate(answer.split()):
            jitter = ((i * 2654435761) % 100) / 100 * 0.3
            out.append(round(base - jitter, 4))
        return out or [math.log(max(conf, 1e-3))]
