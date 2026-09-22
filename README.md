# LocalJev

**A local benchmark that shows *why* [System One models / Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) win — running entirely on [LM Studio](https://lmstudio.ai/).**

TypeSafe's **Jev** is a hosted "System One" model: *unstructured state in, typed
probabilistic decisions out.* Instead of generating text it answers pre-defined
typed questions and returns **calibrated probabilities + confidence**, fast, cheap,
and with no hallucinated categories.

LocalJev reproduces that developer experience **100% locally** (Jev's exact
[`/v1/systemone` contract](https://docs.typesafe.ai/api.md): Choice · Score · Noul)
**and then measures the advantages** by running the *same LM Studio model* two ways:

- **System One path** — constrained to a single answer-token, real probabilities read from the token **logprobs**.
- **Free-form LLM path** — the classic approach: prompt the model to emit JSON, then parse it.

On the same model, the benchmark scores both against a labeled dataset:

| Metric (bonsai-8b, 8 tickets) | System One | Free-form LLM |
|---|---|---|
| **Accuracy** vs ground truth | **0.88** | 0.88 |
| **Output tokens / decision** (cost driver) | **1.0** | ~20 |
| **Calibration** — Brier (lower better) | **0.12** | 0.21 |
| **Type-safe outputs** | **100%** | 100%* |
| Latency / ticket | ~350 ms | ~570 ms |
| **Projected @ 1M decisions** | **1.0M output tokens** | ~20M output tokens |

\* type-safety is *guaranteed* for System One and *observed* for the baseline — weaker
models drop below 100% (invalid JSON / hallucinated categories); System One never can.

The dashboard streams these scorecards in live as it runs, with per-metric "win"
badges and a 1M-decision cost projection.

## Why the numbers come out this way

- **Output-token cost.** System One emits **1 token per decision**; a free-form model
  spends 20–150 tokens per ticket writing JSON. Output tokens are what you pay for —
  hence the "Jevons efficiency" pitch. Jev itself charges **$0 for output tokens**.
- **Real probabilities, not vibes.** Asking a model to *write* `"confidence": 0.9`
  gives a made-up number. LocalJev reads the model's own token distribution from
  `logprobs`, so the probabilities are meaningful — reflected in a **lower Brier score**.
- **Type-safety by construction.** The answer can only ever be a declared label, so
  invalid/hallucinated categories are structurally impossible. The baseline has to
  parse free text and sometimes fails.
- **Latency.** A constrained 1-token decision is cheap; a real System One model
  (parallel sampler) is 40–200× faster than a frontier LLM per TypeSafe. Locally on one
  llama.cpp instance the gap is smaller, but System One still usually wins wall time.

## The three primitives

| Primitive | Question | Returns |
|-----------|----------|---------|
| **Choice** | pick one of N options | `choice` + `probabilities` per option + `confidence` |
| **Score**  | rate on ordered levels | continuous `score` + per-level `probabilities` + `confidence` |
| **Noul**   | yes / no | `noul` = P(yes) + `confidence` |

## Quick start

**1. LM Studio** — Developer tab → load a **plain instruct GGUF** (see model
requirements below) → **Start Server** (defaults to `http://localhost:1234`).

> ### ⚠ Model requirements (read this if you hit a logprobs error)
> LocalJev reads the probability of the *answer token*, so the model must
> **return token logprobs** (llama.cpp **GGUF** models do; some MLX builds don't) and
> **not be a reasoning/"thinking" model** (those spend the first tokens in a hidden
> channel, leaving no answer token — you'll get an actionable error).
>
> **Good picks:** Llama-3.x-Instruct, Qwen2.5-Instruct (non-thinking), Gemma-2-it,
> Phi-3.5-mini, or any small instruct GGUF. LocalJev auto-selects a *loaded* text LLM
> (via LM Studio's `/api/v0/models`) and the dashboard dropdown groups Loaded vs
> Available (● = loaded). Pin one with `LOCALJEV_MODEL`.

**2. Install + run:**

```bash
pip install -r requirements.txt
./run.sh                 # or: python -m localjev.server
```

**3. Open the dashboard** at <http://localhost:8000>, pick a model, and hit
**Run benchmark**. Or from the terminal:

```bash
make bench MODEL=bonsai-8b LIMIT=8     # System One vs LLM scorecard
```

## Using the decision API directly

```bash
curl -s http://localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "questions": {
    "department":  {"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments, refunds","technical":"Bugs, outages","sales":"Pricing, upgrades"}},
    "frustration": {"type":"score","instructions":"How frustrated?","criteria":["Calm","Frustrated","Very angry"]},
    "is_urgent":   {"type":"noul","instructions":"Urgent?","criteria":{"true":"Time-sensitive","false":"Not urgent"}}
  }
}' | python -m json.tool
```

```python
from localjev.sdk import LocalJev, choice, score, noul

jev = LocalJev()
r = jev.evaluate("You charged me twice. I want a refund now.", {
    "department":   choice("Which team?", {"billing":"…","technical":"…","sales":"…"}),
    "frustration":  score("How frustrated?", ["Calm","Frustrated","Very angry"]),
    "wants_refund": noul("Refund requested?", true="asks for money back", false="no refund"),
})
print(r["answers"]["department"]["choice"], r["answers"]["department"]["probabilities"])
```

## How the benchmark works

```
  labeled tickets (data/tickets.jsonl)
        │
        ├──▶ System One:  each question → 1-token constrained call → logprobs → distribution
        │
        └──▶ Free-form:   one call → generate JSON → parse → coerce to labels
        │
        ▼
  score both vs ground truth:
    accuracy · output tokens · latency · type-safety · Brier (calibration)
        │
        ▼
  stream scorecards + 1M-decision projection to the dashboard (SSE)
```

## Endpoints

| Route | Purpose |
|-------|---------|
| `POST /v1/systemone` | Jev-shaped typed decisions (Choice/Score/Noul) |
| `POST /v1/benchmark` | streaming (SSE) System One vs free-form-LLM benchmark |
| `GET /health`, `GET /v1/models` | LM Studio connectivity + model catalog (load state/type) |
| `GET /` | the benchmark + triage dashboard |

## Layout

```
localjev/
  lmstudio.py   # only thing that talks to LM Studio: logprobs + free-form generate
  engine.py     # System One path: logprobs -> Choice/Score/Noul, calibration, confidence
  baseline.py   # free-form LLM path: generate JSON, parse, coerce, flag invalid
  metrics.py    # pure scoring: Brier, accuracy, summaries, projection
  benchmark.py  # runs both over data/tickets.jsonl, streams scorecards
  server.py     # FastAPI: /v1/systemone, /v1/benchmark, /health, dashboard
  sdk.py        # tiny Python client + choice()/score()/noul() builders
web/index.html  # benchmark dashboard (+ "Try one ticket" triage tab)
data/tickets.jsonl   # labeled dataset (department / is_urgent / wants_refund)
examples/       # triage.py, benchmark_cli.py
tests/          # 39 tests, LM Studio mocked (make test)
```

## Development

```bash
make install-dev
make test          # 39 tests, no LM Studio required
make bench         # terminal benchmark (needs LM Studio)
make run           # server + dashboard
make help
```

## Config (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `LOCALJEV_LMSTUDIO_URL` | `http://localhost:1234/v1` | LM Studio OpenAI endpoint |
| `LOCALJEV_MODEL` | *(auto: a loaded text LLM)* | pin a specific model |
| `LOCALJEV_CALIB_T` | `1.0` | calibration temperature on the logits |
| `LOCALJEV_CONCURRENCY` | `1` | parallel questions per ticket (raise only if LM Studio serves concurrently) |
| `LOCALJEV_PORT` | `8000` | LocalJev server port |

## Notes & honesty

- Independent, educational re-implementation of the *System One developer experience*
  and a fair local benchmark of it. **Not** TypeSafe's Jev model; not affiliated with
  TypeSafe AI. Exact numbers depend on your model and hardware.
- The baseline is a genuine, reasonable way to use an LLM (one call, structured JSON) —
  the comparison isn't rigged. System One's structural advantages (1-token cost,
  guaranteed type-safety, logprob-based calibration) are what show up in the scores.
- Jev's headline edge is a model *trained* (RLCD) and served (parallel sampler) for
  calibrated decisions; locally we demonstrate the mechanism and contract.

## License

MIT — see [LICENSE](LICENSE).
