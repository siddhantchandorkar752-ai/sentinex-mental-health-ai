# SENTINEX

> A session-isolated, non-clinical workbench for inspecting language-model labels and literal
> phrase matches.

[![CI](https://github.com/siddhantchandorkar752-ai/sentinex-mental-health-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/siddhantchandorkar752-ai/sentinex-mental-health-ai/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

SENTINEX shows what a classifier returned while making the interpretation boundary visible. It
does **not** diagnose a condition, detect hidden intent, assess crisis risk, recommend treatment,
or determine whether a person is safe.

[Open the Hugging Face Space](https://huggingface.co/spaces/siddhantchandorkar/sentinex-mental-health-ai)
(the Space may sleep when idle). The default deployment runs the synthetic demo backend; it is a
UI and data-flow demonstration, not an AI accuracy demonstration.

## What is verified

- **VERIFIED — deterministic demo:** the default backend is offline, input-dependent, and
  reproducible. Its distributions are synthetic and have no semantic meaning.
- **VERIFIED — session isolation:** each browser session owns its aggregate history. Raw submitted
  text is excluded from that state and from the machine-readable report.
- **VERIFIED — bounded retention:** at most 10 aggregate entries are kept in session state.
- **VERIFIED — transparent lexical rules:** exact categories and literal phrases live in
  [`models/analyzer.py`](models/analyzer.py). Whole-phrase boundaries are tested.
- **VERIFIED — pinned adapters:** optional emotion and sentiment backends use immutable Hugging
  Face model revisions and reject unknown label schemas. Remote model code is disabled.
- **VERIFIED — offline test suite:** model adapters are tested with fakes; CI downloads no model
  weights and uses no credentials.
- **UNVERIFIED — model quality:** this project contains no benchmark establishing accuracy,
  fairness, robustness, crisis detection, or clinical validity.

## Safety boundary

Classifier confidence is confidence in a model label. It is not symptom severity, the probability
of a diagnosis, or a safety measurement. Literal phrase matching cannot understand negation,
quotation, context, intent, or sarcasm.

SENTINEX cannot determine whether anyone is safe. If you may act on thoughts of self-harm or
someone is in immediate danger, contact local emergency services now. In India, use the Ministry
of Health and Family Welfare's official [Tele-MANAS website](https://telemanas.mohfw.gov.in/).
No phone number is copied here because operational contact details should come from the current
official source.

Do not submit private health information, identifying data, or emergency information. The app
does not intentionally persist text, but its hosting platform and network path have independent
policies.

## Architecture

```text
browser session
  ├─ submitted text ──> stateless analyzer ──> structured display
  └─ aggregate state <── labels, scores, match counts, mode (max 10)
                              │
                              ├─ demo (default): deterministic SHA-256 fixture, offline
                              └─ transformers (opt-in): two pinned classifier adapters
```

The UI is import-safe: importing `app` neither starts a server nor initializes models. The
transformer backend is initialized lazily after a request and serializes access to its pipelines.

## Run the verified demo

Requires Python 3.11–3.13.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[ui]"
python app.py
```

Open <http://127.0.0.1:7860>. `SENTINEX_MODE=demo` is the safe default and makes no model-network
requests.

## Optional pinned transformer mode

This mode downloads model weights from Hugging Face and can require substantial memory. Its output
remains non-clinical and unvalidated.

```bash
python -m pip install -e ".[ui,models]"
# Windows PowerShell
$env:SENTINEX_MODE = "transformers"
python app.py
```

Pinned models:

- Emotion: `j-hartmann/emotion-english-distilroberta-base` at
  `0e1cd914e3d46199ed785853e12b57304e04178b`
- Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest` at
  `3216a57f2a0d9c45a2e6c20157c20c49fb4bf9c7`

The previous sarcasm model was removed because its published configuration did not define a label
mapping. Treating generic `LABEL_0`/`LABEL_1` values as “sarcasm” would be an unsupported semantic
claim. Contrast-related wording remains visible only as a literal-match category.

## Test and audit

```bash
python -m pip install -e ".[ui,dev,security]"
python -m ruff format --check .
python -m ruff check .
python -m pytest
python -m pip check
python -m pip_audit --local --skip-editable
```

CI runs lint, compilation, and offline tests on Python 3.11, 3.12, and 3.13; audits installed UI
dependencies; and inspects a source-only wheel. See [`SECURITY.md`](SECURITY.md) for private
reporting guidance.

## Docker

```bash
docker build -t sentinex .
docker run --rm -p 7860:7860 sentinex
```

The image runs as an unprivileged user and defaults to demo mode. Set transformer mode only when
you intentionally provide the required compute and model-download access. Build that larger image
explicitly:

```bash
docker build --build-arg SENTINEX_EXTRAS=ui,models -t sentinex-models .
docker run --rm -e SENTINEX_MODE=transformers -p 7860:7860 sentinex-models
```

## Intended and prohibited use

Intended: education, UI experiments, classifier-adapter testing, and non-clinical NLP research
with synthetic or non-sensitive text.

Prohibited: diagnosis, triage, monitoring a person, emergency response, treatment decisions,
employment or insurance decisions, or any high-stakes judgment about an individual.

## License

[MIT](LICENSE) © 2026 Siddhant Chandorkar.
