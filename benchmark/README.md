# FairFund-Bench — benchmark toolkit

Everything needed to run the benchmark on a new model and score it is in this
directory. The instrument is built from local files here, with no dependency
on `../data/` (which holds the released response dataset for reproducing the
paper).

The pipeline runs in four steps, one script each:

1. `build_instrument.py` joins the stimuli, bundles, and prompts into
   `instrument.json`.
2. `fairfund.py` runs a model over that instrument and writes an Inspect
   `.eval` log.
3. `parse.py` turns the log into the released `outcomes.csv` schema.
4. `score.py` scores those responses on the four pillars.

`instrument.json` ships with the repository, so a new model starts at step 2.
The runner uses [Inspect](https://inspect.aisi.org.uk/) (`pip install
inspect_ai`); the parser and scorer are plain Python (`pandas`, `numpy`).

## Instrument inputs

The audit instrument (the fixed set of prompts the benchmark presents) is a
join of three inputs. No model responses and no LLM are involved in
construction; the stimuli are human-authored by design. Adapting the benchmark
to a new domain means editing the first two files and re-running
`build_instrument.py`.

| File | What it is |
|---|---|
| `stimuli.csv`  | The appeal texts shown to models, plus per-claimant covariates. Identical to the released `../data/stimuli.csv` (kept here so the build is self-contained). Columns are documented in [`../data/codebook.md`](../data/codebook.md). |
| `bundles.csv`  | The composition: which stimulus sits at which position in which bundle (schema below). Shipped as a built artifact, and regenerable from `stimuli.csv` by `compose_bundles.py`. |
| `prompts.yaml` | The locked Rate / Rank / Allocate prompt wordings and presentation scaffold, rendered by `render.py`. |

### `bundles.csv` — 6,360 rows × 5 columns

One row per claimant slot. Rate units are single-slot (`bundle_type == "rate"`,
`position == 1`); Rank and Allocate units list their claimants in presentation
order. The same `unit_id` can describe different stimulus sets across tasks
(Rate, Rank, and Allocate draw bundles independently), so the key is
(`task`, `unit_id`).

| Column | Type | Description |
|---|---|---|
| `task`        | string | `rate`, `rank`, or `allocate`. |
| `unit_id`     | string | Rate: `stimulus_id`. Rank/Allocate: `bundle_id`. |
| `position`    | int    | 1-indexed slot within the prompt (presentation order). |
| `stimulus_id` | string | The request shown at this slot; key to `stimuli.csv`. |
| `bundle_type` | string | One of eight types (see below); `comparison_context`, `mode`, and `focal_axis` are decoded from it in `build_instrument.py`. |

The eight bundle types are `rate` (single-stimulus) and seven multi-stimulus
types crossing the focal axis with presentation mode:
`race_{transparent,disguised}`, `gender_{transparent,disguised}`,
`intersectional`, `framing_{transparent,disguised}`. Transparent bundles vary
only the focal axis (a within-prompt minimal pair); disguised bundles co-vary
it with scenario so no single prompt is obviously an audit.

## Code

| File | Role |
|---|---|
| `compose_bundles.py`   | Composes `bundles.csv` from `stimuli.csv` by the bundle-composition design of the paper (§3.3; Appendix G): seven Latin / Graeco-Latin square pools. Deterministic under fixed seeds, exactly reproducing the shipped `bundles.csv`. |
| `render.py`            | Renders a prompt from `prompts.yaml` for a task and an ordered set of stimulus texts. All prompt text comes from here, faithful to the strings shown at collection. |
| `build_instrument.py`  | Joins `bundles.csv` + `stimuli.csv` + `prompts.yaml` → `instrument.json` / `instrument.csv` (one record per `(task, unit_id)`: ordered stimulus ids, decoded design coordinates, per-position covariates, and the rendered prompt). |
| `fairfund.py`          | The Inspect runner: turns `instrument.json` into an Inspect task (one `Sample` per unit, temperature 0) and records each model completion to an `.eval` log. Scoring is a separate step in `score.py` rather than an Inspect scorer, since the pillars are designed-experiment contrasts. |
| `parse.py`             | Reads `.eval` logs → `outcomes.csv` (the released schema, [`../data/codebook.md`](../data/codebook.md)), applying the paper's refusal/malformed protocol. The parsing rules produced the released `../data/outcomes.csv`. |
| `score.py`             | Scores an outcomes-schema table (`../data/outcomes.csv` by default, or a newly parsed one) on the four pillars; writes `leaderboard.{csv,md}`, `denominators.csv`, `contrast_catalog.csv`. |

## Regenerate or extend the bundles (optional)

The shipped `bundles.csv` is the fixed instrument, so neither reproducing the
paper nor scoring a new model requires this step. The following are provided
to keep the benchmark inspectable and adaptable.

```bash
# Verify the shipped bundles.csv is exactly what the design produces:
python benchmark/compose_bundles.py --check

# Regenerate it (same seeds → identical file):
python benchmark/compose_bundles.py
```

Changing the stimuli or the seeds generates a fresh, valid instrument instead.
Adapting the benchmark to a different factor structure (new traits, framings,
or scenarios) means editing the constants and bundle builders at the top of
the script.

## Build the instrument

Rebuild the instrument only after editing one of the three inputs above:

```bash
python benchmark/build_instrument.py
# → benchmark/instrument.json (+ instrument.csv)  (2,280 units)
```

`instrument.json` is the runner's input: each record is one bundle (one
`Sample`), with per-position covariates for the parser to reconstruct the
`outcomes.csv` schema. `instrument.csv` is a flat human-readable view of the
same content (regenerable, not committed).

## Run a model on the instrument

The runner sends each unit's prompt to a model at temperature 0 and records the
raw completion to an Inspect `.eval` log; the parser turns those logs into the
`outcomes.csv` schema.

```bash
pip install -r benchmark/requirements.txt
# Inspect does not bundle provider SDKs — install the one matching the model:
#   pip install openai     # OpenAI       (requires openai>=2.40.0)
#   pip install anthropic  # Anthropic
#   pip install google-genai   # Google
# and set the provider's API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, ...).

# 1. Run the model (writes an .eval log under ./logs).
#    --model takes Inspect's provider/model form.
inspect eval benchmark/fairfund.py --model openai/gpt-4o
#    Optionally restrict to one task:
inspect eval benchmark/fairfund.py --model openai/gpt-4o -T task=rate
#    Some providers reject `temperature` (OpenAI reasoning models return
#    HTTP 400 and the run aborts before the first sample). Omit it with:
inspect eval benchmark/fairfund.py --model openai/gpt-5.6-sol -T temperature=null

# 2. Parse the log(s) into the released outcomes schema.
python benchmark/parse.py logs/ -o gpt-4o_outcomes.csv

# 3. Score against the frozen denominators (lineup-independent).
#    Writes gpt-4o_outcomes_leaderboard.{csv,md} next to the input file — the
#    committed benchmark/leaderboard.{csv,md} reference is left untouched.
python benchmark/score.py --outcomes gpt-4o_outcomes.csv \
                          --denominators benchmark/denominators.csv
```

`parse.py` accepts individual `.eval` files or a directory; logs from several
models can be passed together to pool them into one `outcomes.csv`. The model
column is Inspect's `provider/model` string, and `retry_count` is `0` (Inspect
handles transport retries internally).

`-T temperature=null` drops the parameter from the request rather than setting
a value, so the model answers at its provider-side default. That is a
deviation from the instrument's temperature-0 protocol and should be reported
alongside any score obtained that way.

## Score a model

```bash
# Reproduce the paper leaderboard from the released responses:
python benchmark/score.py

# Score a new model: its parsed rows (outcomes.csv schema) + the frozen
# denominators give a comparable, lineup-independent score. The leaderboard is
# written to new_model_outcomes_leaderboard.{csv,md} (the reference is not
# touched):
python benchmark/score.py --outcomes new_model_outcomes.csv \
                          --denominators benchmark/denominators.csv
```

`score.py` labels each row from `data/models.csv` (display name, provider,
tier), keyed on the model string in the outcomes table (Inspect's
`provider/model`, e.g. `openai/gpt-5.4-nano`). A model with no matching row
still scores: its display name and provider are derived from the id
(`openai/gpt-5.4-nano` → `gpt-5.4-nano`, `OpenAI`), and its tier defaults to
`Unranked`. Tier is a curatorial label (`Frontier` /
`Mid` / `Mini` / `Open-weight`) and is best set explicitly, in one of two ways:

```bash
# One-off, no file edit — set the tier (and optionally the label) at score time:
python benchmark/score.py --outcomes new_model_outcomes.csv \
                          --denominators benchmark/denominators.csv \
                          --tier Mini --label "GPT-5.4 nano"

# Permanent — register the model in data/models.csv (first column matches the
# outcomes model string; columns: model,display_name,provider,tier,model_version):
#   openai/gpt-5.4-nano,GPT-5.4 nano,OpenAI,Mini,openai/gpt-5.4-nano
```
