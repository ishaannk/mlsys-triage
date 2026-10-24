# Triage campaign.
#
# The only target that matters before renting anything is `dryrun`. When it
# passes, only compute remains.

PY ?= python
OUT ?= runs

.PHONY: help test dryrun bringup solo sampler price data clean app-test all-test campaign

help:
	@echo "make test          - campaign unit tests (CPU, no GPU, no network)"
	@echo "make app-test      - the app repo's own tests"
	@echo "make all-test      - both"
	@echo "make dryrun        - THE GATE: T1-T7 + F1-F6 on the mock substrate"
	@echo "make data          - pre-download the primary datasets onto the volume"
	@echo ""
	@echo "on the pod, in this order:"
	@echo "  make solo        - BEFORE any server starts: is the card really ours?"
	@echo "  make bringup     - S0 gate: tiers, metrics, SSE, noise floor, resume"
	@echo "  make sampler     - NVML sampler, separate process, leave it running"
	@echo "  make price       - S5+S6: measure the price of one escalation (C2)"
	@echo "  make campaign    - the real headline run on the primary mix"

test:
	$(PY) -m pytest tests/ -q

app-test:
	cd ../triage && $(PY) -m pytest -q

all-test: test app-test

dryrun:
	$(PY) -m tri.dryrun --n 40 --seeds 2 --out $(OUT)/dryrun

# Pull the four primary datasets onto the network volume BEFORE any timed stage.
# A first-request-triggered download inside a timed window is a 20-minute outlier.
data:
	$(PY) -c "from tri.eval.datasets import load_primary; \
	          items = load_primary(); \
	          print('cached %d items across the primary mix' % len(items))"

solo:
	$(PY) -m tri.sampler --solo-check

bringup:
	$(PY) scripts/bringup.py --noise-n 100

sampler:
	TRIAGE_RUN_ID=$${TRIAGE_RUN_ID:-adhoc} $(PY) -m tri.sampler --out $(OUT)/gpu_samples.jsonl --hz 10

# S5 (headroom) + S6 (load). Writes runs/price_rows.json, which T4/F1/F2 read.
price:
	$(PY) scripts/probe_price.py --mode both --n 400 --seeds 3

# S12 + N5: the counterfactual log and the learning stack it unblocks.
cflog:
	$(PY) -u -m tri.campaign --items primary --substrate mock --cflog-n 400 	  --loads 12 --bisect-n 120 --out $(OUT)/cflog

campaign:
	$(PY) -u -m tri.campaign --items primary --substrate vllm_local --seeds 3 	  --bisect-n 250 --corrupt-n 100 --loads 2,6,12,20 --out $(OUT)/headline

clean:
	rm -rf $(OUT)/dryrun
