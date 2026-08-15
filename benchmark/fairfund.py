"""
Program:   fairfund.py
Task:      The FairFund-Bench Inspect runner (step 2 of the pipeline in
           benchmark/README.md). Turns the fixed instrument
           (benchmark/instrument.json) into an Inspect evaluation that
           presents each unit's prompt to a model and records the raw
           completion. One Sample per (task, unit_id); only the model's
           text answer is retained. Scoring is done in the standalone
           benchmark/score.py, whose pillars are estimated experimental contrasts.

           Run a model, then parse the log into the released outcomes schema:

             inspect eval benchmark/fairfund.py --model openai/gpt-4o
             python benchmark/parse.py logs/ -o gpt-4o_outcomes.csv
             python benchmark/score.py --outcomes gpt-4o_outcomes.csv

           The instrument is held at temperature 0 (the paper's setting for
           all three tasks). Optionally restrict to one task:

             inspect eval benchmark/fairfund.py --model openai/gpt-4o -T task=rate

           Some providers reject `temperature` outright (OpenAI reasoning
           models return HTTP 400). Omit the parameter for those, and note
           in any writeup that the run is at the provider default, not 0:

             inspect eval benchmark/fairfund.py --model openai/gpt-5.6-sol \
                 -T temperature=null

Inputs:    benchmark/instrument.json  (build with build_instrument.py)
Outputs:   an Inspect .eval log under ./logs (parsed by benchmark/parse.py)

Project:   FairFund-Bench
Author:    Martin Lukk
"""

# 0. Program Setup --------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import generate

SCRIPT_DIR    = Path(__file__).resolve().parent
INSTRUMENT    = SCRIPT_DIR / "instrument.json"
TASKS         = ("rate", "rank", "allocate")

# Everything parse.py needs to rebuild an outcome row.
_METADATA_KEYS = ("task", "unit_id", "wording_variant", "bundle_type",
                  "comparison_context", "mode", "focal_axis", "bundle_size",
                  "stimulus_ids", "positions")


# 1. Instrument -> Samples ------------------------------------------------
def _load_samples(task_filter: str | None) -> list[Sample]:
    if not INSTRUMENT.exists():
        raise FileNotFoundError(
            f"{INSTRUMENT} not found. Build it first:\n"
            f"    python benchmark/build_instrument.py")
    records = json.loads(INSTRUMENT.read_text())

    samples: list[Sample] = []
    for r in records:
        if task_filter and r["task"] != task_filter:
            continue
        samples.append(Sample(
            input=r["prompt"],
            id=f'{r["task"]}__{r["unit_id"]}',
            metadata={k: r[k] for k in _METADATA_KEYS},
        ))
    if not samples:
        raise SystemExit(
            f"No units selected (task={task_filter!r}). "
            f"Expected one of {TASKS} or None.")
    return samples


# 2. Task -----------------------------------------------------------------
@task
def fairfund(task: str | None = None,
             temperature: float | None = 0.0) -> Task:
    """FairFund-Bench instrument. `task` optionally restricts to one of
    rate / rank / allocate (default: all three). `temperature` is the
    instrument's setting (0); pass `-T temperature=null` to omit the
    parameter for models that reject it (e.g. OpenAI reasoning models)."""
    if task is not None and task not in TASKS:
        raise SystemExit(f"task must be one of {TASKS}, got {task!r}.")
    return Task(
        dataset=MemoryDataset(_load_samples(task)),
        solver=generate(),
        # No scorer: completions are recorded and scored by benchmark/score.py.
        # temperature=None is dropped from the request entirely.
        config=GenerateConfig(temperature=temperature),
    )
