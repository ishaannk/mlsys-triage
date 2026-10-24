"""Calibrate verify.overhead_tokens from real telemetry — so the verify cost
guard uses a MEASURED overhead instead of a magic constant.

The verify pass adds a second LLM call. Its overhead is estimated as the
difference in median total tokens (in+out) between requests that ran a verify
pass and those that did not. This is a telemetry estimate (verified requests are
the harder ones, so treat it as an upper-ish bound), printed as a recommended
value for `verify.overhead_tokens` in config/router.yaml.

  python -m app.benchmark.verify_overhead
"""
from __future__ import annotations

import statistics as st

from ..telemetry.db import _conn


def measure() -> dict:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT verified, tokens_in, tokens_out FROM requests "
            "WHERE tokens_in IS NOT NULL AND tokens_out IS NOT NULL"
        ).fetchall()
    verified = [r["tokens_in"] + r["tokens_out"] for r in rows if r["verified"]]
    plain = [r["tokens_in"] + r["tokens_out"] for r in rows if not r["verified"]]
    result = {
        "n_verified": len(verified),
        "n_plain": len(plain),
        "median_total_verified": round(st.median(verified), 1) if verified else None,
        "median_total_plain": round(st.median(plain), 1) if plain else None,
    }
    if verified and plain:
        result["measured_overhead_tokens"] = round(
            st.median(verified) - st.median(plain), 1
        )
    return result


def main() -> None:
    r = measure()
    print("[verify_overhead] from telemetry.db")
    for k, v in r.items():
        print(f"  {k}: {v}")
    ov = r.get("measured_overhead_tokens")
    if ov is None:
        print("  Not enough verified/plain samples yet — keep the configured default.")
    else:
        print(f"\n  -> set verify.overhead_tokens: {max(0, int(ov))} in config/router.yaml")


if __name__ == "__main__":
    main()
