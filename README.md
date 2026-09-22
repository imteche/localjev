# LocalJev

**A self-hosted [System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev) decision engine that runs on any model in [LM Studio](https://lmstudio.ai/).**

TypeSafe's **Jev** is a hosted "System One" model — *unstructured state in, typed
probabilistic decisions out.* Instead of generating text, it answers pre-defined
typed questions and returns **calibrated probabilities + confidence**, with no
hallucinated categories. LocalJev reproduces that developer experience **100%
locally**: it speaks Jev's exact [`/v1/systemone` contract](https://docs.typesafe.ai/api.md)
(Choice · Score · Noul) but is backed by a model you run in LM Studio.

### Why this isn't just "ask the LLM for JSON"

Asking a model to *write* `"confidence": 0.88` gives you a made-up number. LocalJev
instead does what a real System One model does — it reads probabilities out of the
model's own token distribution:

1. **Real probabilities from logprobs.** Each question is compiled to a
   single-token answer (a digit per label). LocalJev reads LM Studio's
   `top_logprobs` for those tokens and softmaxes them into a genuine distribution
   over the declared labels. Score returns the probability-weighted expected level
   (e.g. `1.05`), exactly like Jev.
2. **Type-safety by construction.** The answer can only ever be one of the labels
   you declared — hallucinated categories are structurally impossible.
3. **Calibration + confidence + escalation.** Confidence = `1 − normalized entropy`
   of the distribution; a temperature-scaling knob (`LOCALJEV_CALIB_T`) tunes
   calibration; low-confidence decisions auto-route to a human.

## The three primitives

| Primitive | Question | Returns |
|-----------|----------|---------|
| **Choice** | pick one of N options | `choice` + `probabilities` per option + `confidence` |
| **Score**  | rate on ordered levels | continuous `score` + per-level `probabilities` + `confidence` |
| **Noul**   | yes / no | `noul` = P(yes) + `confidence` |

## Quick start

**1. LM Studio** — open the **Developer** tab, load a **plain instruct GGUF**
(see model requirements below), and **Start Server** (defaults to
`http://localhost:1234`).

> ### ⚠ Model requirements (read this if you hit a logprobs error)
>
> LocalJev reads the probability of the *answer token*, so it needs a model that:
> 1. **returns token logprobs** — llama.cpp **GGUF** models in LM Studio do; some
>    MLX builds do not; and
> 2. **is not a reasoning / "thinking" model.** Thinking models (e.g. QwQ, DeepSeek-R1
>    distills, Qwen3 in thinking mode, phi-4-reasoning) spend their first tokens in a
>    hidden reasoning channel, so there is *no answer token to read* — you'll get
>    `content: ""` and `logprobs: null`, and LocalJev will tell you so.
>
> **Good picks:** Llama-3.x-Instruct, Qwen2.5-Instruct (non-thinking), Gemma-2-it,
> Phi-3.5-mini, or any small instruct GGUF.
>
> **Auto-selection:** LocalJev queries LM Studio's native `/api/v0/models` and
> defaults to a model that is *actually loaded* and is a text LLM — it will **not**
> silently JIT-load a huge not-loaded vision model just because it's listed first.
> The dashboard dropdown groups **Loaded** vs **Available** models (● = loaded) so
> you can switch without restarting; pin one server-side with `LOCALJEV_MODEL`.

**2. Install + run:**

```bash
pip install -r requirements.txt
./run.sh                 # or: python -m localjev.server
```

**3. Open the dashboard** at <http://localhost:8000> and evaluate a ticket. Or hit
the API directly:

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

Response (Jev-shaped):

```json
{
  "model": "…",
  "answers": {
    "department":  {"type":"choice","choice":"billing","probabilities":{"billing":0.88,"technical":0.12,"sales":0.0},"confidence":0.79},
    "frustration": {"type":"score","score":1.05,"legend":{"0":"Calm","1":"Frustrated","2":"Very angry"},"probabilities":{"0":0.0,"1":0.95,"2":0.05},"confidence":0.83},
    "is_urgent":   {"type":"noul","noul":0.95,"confidence":0.95}
  },
  "usage": {"input_tokens": 307, "output_tokens": 4},
  "latency_ms": 214.6
}
```

## Python SDK

```python
from localjev.sdk import LocalJev, choice, score, noul

jev = LocalJev()
r = jev.evaluate(
    state="You charged me twice. I want a refund now.",
    questions={
        "department":  choice("Which team?", {"billing":"…","technical":"…","sales":"…"}),
        "frustration": score("How frustrated?", ["Calm","Frustrated","Very angry"]),
        "wants_refund": noul("Refund requested?", true="asks for money back", false="no refund"),
    },
)
print(r["answers"]["department"]["choice"])        # -> "billing"
print(r["answers"]["department"]["probabilities"])  # real distribution from logprobs
```

CLI demo over a few sample tickets:

```bash
python examples/triage.py
python examples/triage.py "The API is down and I'm losing money!!"
```

## How it works

```
  ticket text ──▶ LocalJev engine ──▶ compile each question to a 1-token
                                       (digit-per-label) prompt
                        │
                        ▼
                 LM Studio  /v1/chat/completions   (max_tokens=1, logprobs=true)
                        │
                        ▼
              read top_logprobs for the label digits
                        │
              softmax (with calibration T) ──▶ distribution over labels
                        │
        ┌───────────────┼────────────────┐
     Choice           Score             Noul
   argmax + dist   E[level] + dist    P(yes) + conf
```

## Development

```bash
make install-dev   # runtime + test deps
make test          # 15 tests, no LM Studio required (mocks the logprob backend)
make run           # start server + dashboard
make help          # list all tasks
```

The suite covers the engine math (distribution recovery from logprobs, expected-value
scoring, confidence vs. entropy, type-safe off-menu fallback) and the HTTP contract
(`/v1/systemone` Jev shape, validation errors, `/health`, dashboard) — all with LM
Studio mocked, so `make test` runs offline.

## Layout

```
localjev/
  lmstudio.py   # the only thing that talks to LM Studio; pulls token logprobs
  engine.py     # logprobs -> Choice / Score / Noul, calibration, confidence
  server.py     # FastAPI: POST /v1/systemone, /health, and the dashboard
  sdk.py        # tiny Python client + choice()/score()/noul() builders
web/index.html  # live support-triage dashboard
examples/triage.py
tests/          # pytest suite (engine math + HTTP contract), LM Studio mocked
Makefile        # make install-dev / test / run / demo / health
```

## Config (env vars)

| Var | Default | Meaning |
|-----|---------|---------|
| `LOCALJEV_LMSTUDIO_URL` | `http://localhost:1234/v1` | LM Studio OpenAI endpoint |
| `LOCALJEV_MODEL` | *(auto: a **loaded** text LLM)* | pin a specific model |
| `LOCALJEV_CALIB_T` | `1.0` | calibration temperature on the logits |
| `LOCALJEV_PORT` | `8000` | LocalJev server port |

## Notes & honesty

- LocalJev is an independent, educational re-implementation of the *System One
  developer experience*. It is **not** TypeSafe's Jev model and is not affiliated
  with TypeSafe AI. Speed/quality depend on your local model and hardware.
- Local calibration is only as good as the base model plus the `CALIB_T` knob —
  Jev's advantage is a model *trained* (RLCD) for calibrated decisions. The point
  here is to show the mechanism and contract locally.
- Requires a model/runtime that returns token logprobs (llama.cpp GGUF models in
  LM Studio do).

## License

MIT — see [LICENSE](LICENSE).
