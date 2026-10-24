"""S12: the counterfactual log. Plan section 4 - "the single best-value line".

For every item, execute EVERY action arm and grade the outcome. That converts a
bandit problem (you only see what you chose) into a full-information one (you see
what every choice would have given), which is what makes the entire learning
track - repairability heads, action-value heads, off-policy evaluation -
buildable OFFLINE, months later, without renting anything again.

Two disciplines make the log trustworthy:

1. **Shared prefix, by design.** Every arm starts from the SAME small-tier main
   pass, so the arms differ only in what was bought afterwards. That is what
   makes the comparison paired and cheap.

2. **The costs in this log are NOT deployment costs**, and the schema says so in
   every row (`cost_is_counterfactual: true`). Sharing a prefix across six arms
   makes each arm look cheaper than it would be alone, and the arms were never
   under realistic contention. **Use this log for QUALITY LABELS only.** Every
   cost number in the paper comes from a deployment run. Plan section 9.2 #13.

The six arms are the plan's six actions, expressed as what a policy could
actually choose after seeing the small model's answer.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from . import actions as A
from .actions import Ctx, Item
from .eval.graders import grade
from .ledger import Ledger, key_of
from .types import mono

# Each arm is the action sequence a policy could take AFTER the shared main pass.
ARMS: dict[str, list[str]] = {
    "none":            [],
    "resample":        ["resample"],
    "selfcheck":       ["selfcheck"],
    "retrieve":        ["retrieve"],
    "retrieve_verify": ["retrieve", "verify"],
    "escalate":        ["escalate"],
}


@dataclass
class ArmOutcome:
    item_id: str
    arm: str
    seed: int
    pair_id: str
    dataset: str
    answer: str = ""
    correct: Optional[bool] = None
    risk: float = 0.0
    abstained: bool = False
    signals: dict[str, float] = field(default_factory=dict)        # POST-arm
    # The decision features: what was knowable BEFORE the arm was chosen, i.e.
    # the shared main pass and nothing else. Conditioning a router on post-arm
    # state leaks the action into its own inputs, which silently inflates every
    # head's AUC and biases every OPE estimate built on them.
    prefix_signals: dict[str, float] = field(default_factory=dict)
    prefix_risk: float = 0.0
    actions: list[str] = field(default_factory=list)
    n_calls: int = 0
    gpu_s_marginal: float = 0.0      # cost of the ARM only, excluding the prefix
    gpu_s_with_prefix: float = 0.0
    tokens_out: int = 0
    corruption: str = "none"
    corruption_rate: float = 0.0
    # the honesty flag, in every row
    cost_is_counterfactual: bool = True
    cost_note: str = ("shared main-pass prefix across arms, measured without "
                      "realistic contention; use for QUALITY LABELS only")


async def run_item(item: Item, substrate: Any, cfg: dict[str, Any], seed: int = 0,
                   pair_id: str = "A", arms: Sequence[str] = tuple(ARMS)) -> list[ArmOutcome]:
    """One item, every arm, from a shared main pass."""
    import copy

    from app.router.abstain import abstain_risk

    base = Ctx(item=item, substrate=substrate, cfg=cfg, seed=seed, pair_id=pair_id)
    await A.apply(base, "answer_small")
    prefix_gpu_s = base.act_gpu_s + base.probe_gpu_s
    bs = base.signals
    prefix_signals = {"uncertainty": bs.uncertainty, "instability": bs.instability,
                      "contradiction": bs.contradiction,
                      "retrieval_disagreement": bs.retrieval_disagreement,
                      "evidence_sufficiency": bs.evidence_sufficiency,
                      "retrieval_support": bs.retrieval_support}
    prefix_risk = round(abstain_risk(bs, cfg["abstain"])[0], 5)
    prefix_tokens = sum(c.output_tokens for c in base.calls)

    out: list[ArmOutcome] = []
    for arm in arms:
        # Fork the post-prefix state. The prefix itself is never re-executed:
        # that is the whole cost saving, and the reason the costs are marked
        # counterfactual.
        ctx = Ctx(item=item, substrate=substrate, cfg=cfg, seed=seed, pair_id=pair_id)
        ctx.answer = base.answer
        ctx.main = base.main
        ctx.signals = copy.deepcopy(base.signals)
        ctx.models_used = list(base.models_used)
        ctx.actions_taken = list(base.actions_taken)
        ctx.act_gpu_s = base.act_gpu_s
        ctx.probe_gpu_s = base.probe_gpu_s
        ctx.calls = list(base.calls)

        for step in ARMS[arm]:
            await A.apply(ctx, step)

        from app.router.abstain import should_abstain
        flag, risk, _ = should_abstain(ctx.signals, cfg["abstain"])
        if A.degenerate(ctx.answer):
            flag = True
        ok, _d = (False, {}) if flag else grade(ctx.answer, item.gold, item.grader)
        total = ctx.act_gpu_s + ctx.probe_gpu_s
        s = ctx.signals
        out.append(ArmOutcome(
            item_id=item.item_id, arm=arm, seed=seed, pair_id=pair_id,
            dataset=item.dataset, answer="" if flag else ctx.answer,
            correct=bool(ok), risk=round(risk, 5), abstained=bool(flag),
            signals={"uncertainty": s.uncertainty, "instability": s.instability,
                     "contradiction": s.contradiction,
                     "retrieval_disagreement": s.retrieval_disagreement,
                     "evidence_sufficiency": s.evidence_sufficiency,
                     "retrieval_support": s.retrieval_support},
            prefix_signals=dict(prefix_signals), prefix_risk=prefix_risk,
            actions=list(ARMS[arm]), n_calls=len(ctx.calls),
            gpu_s_marginal=round(total - prefix_gpu_s, 6),
            gpu_s_with_prefix=round(total, 6),
            tokens_out=sum(c.output_tokens for c in ctx.calls),
            corruption=item.corruption, corruption_rate=item.corruption_rate))
        out[-1].signals["_prefix_tokens_out"] = float(prefix_tokens)
    return out


async def build_log(items: Sequence[Item], substrate: Any, cfg: dict[str, Any],
                    out_path: str, seeds: Sequence[int] = (0,), pair_id: str = "A",
                    arms: Sequence[str] = tuple(ARMS)) -> dict[str, Any]:
    """Write counterfactual.jsonl. Resumable on the same key discipline as every
    other stage: (item_id, arm, seed, pair_id, config_hash)."""
    chash = cfg.get("config_hash", "")
    led = Ledger(out_path, schema="counterfactual",
                 flush_every_s=cfg["run"]["flush_every_s"])
    n_written = 0
    t0 = mono()
    try:
        for seed in seeds:
            for it in items:
                if all(led.has(key_of(it.item_id, a, seed, pair_id, chash)) for a in arms):
                    continue
                for o in await run_item(it, substrate, cfg, seed, pair_id, arms):
                    if led.write(key_of(o.item_id, o.arm, o.seed, o.pair_id, chash),
                                 dataclasses.asdict(o)):
                        n_written += 1
    finally:
        led.close()
    return {"rows_written": n_written, "rows_total": len(led.seen),
            "items": len(items), "arms": list(arms), "seeds": list(seeds),
            "wall_s": round(mono() - t0, 2), "path": str(out_path),
            "WARNING": "costs in this log are counterfactual (shared prefix, no "
                       "contention). Quality labels only."}


# --------------------------------------------------------------------------- #
def oracle_policy_value(rows: Sequence[dict], key: str = "correct") -> dict[str, Any]:
    """The best achievable value, and each fixed arm's value.

    Because the log is full-information, the value of ANY deterministic policy
    over these arms is computable exactly. That is what makes the OPE validation
    in `tri.learning` a real check rather than a second implementation of the
    same estimator.
    """
    by_item: dict[tuple, dict[str, dict]] = {}
    for r in rows:
        by_item.setdefault((r["item_id"], r["seed"], r["pair_id"]), {})[r["arm"]] = r
    arms = sorted({r["arm"] for r in rows})
    per_arm = {a: [] for a in arms}
    best = []
    for arms_of_item in by_item.values():
        for a in arms:
            if a in arms_of_item:
                per_arm[a].append(float(bool(arms_of_item[a][key])))
        best.append(max(float(bool(v[key])) for v in arms_of_item.values()))
    return {"n_items": len(by_item),
            "per_arm": {a: round(sum(v) / len(v), 5) for a, v in per_arm.items() if v},
            "oracle": round(sum(best) / len(best), 5) if best else 0.0,
            "note": "oracle knows which arm works for each item; it is an upper "
                    "bound, not a baseline anyone could deploy"}
