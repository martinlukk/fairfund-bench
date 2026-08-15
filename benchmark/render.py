"""
Program:   render.py
Task:      Render Rate / Rank / Allocate prompts from the locked wording in
           benchmark/prompts.yaml. Single source of truth for the prompt
           text: the wording variants (base / need / merit / paraphrase),
           the bundle-size tokens ({n}, {n_word}, {example}), and the
           labeled-prose presentation scaffold all live in the YAML. The
           instrument builder and the Inspect runner both import this module.

Inputs:    benchmark/prompts.yaml   (locked wordings; same file used at
                                      collection time)
Project:   FairFund-Bench
Author:    Martin Lukk
"""

# 0. Program Setup --------------------------------------------------------
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.yaml"

TASKS            = ("rate", "rank", "allocate")
WORDING_VARIANTS = ("base", "need", "merit", "paraphrase")
BUNDLE_SIZES     = (2, 4, 5)


# 1. YAML Loader ----------------------------------------------------------
@lru_cache(maxsize=1)
def _load(path: str = str(PROMPTS_PATH)) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# 2. Instruction Rendering ------------------------------------------------
def _render_instruction(task: str, wording: str, n: int | None) -> str:
    """Substitute bundle-size tokens into the selected wording template."""
    if task not in TASKS:
        raise ValueError(f"unknown task {task!r}; expected one of {TASKS}")
    if wording not in WORDING_VARIANTS:
        raise ValueError(f"unknown wording {wording!r}; expected one of {WORDING_VARIANTS}")

    cfg = _load()
    template = cfg["tasks"][task][wording].rstrip("\n")

    if task == "rate":
        return template

    if n not in BUNDLE_SIZES:
        raise ValueError(f"bundle size {n!r} not in {BUNDLE_SIZES}")

    tokens  = cfg["tokens"]
    example = tokens["example"][task][n]
    n_word  = tokens["n_word"][n]
    return template.format(n=n, n_word=n_word, example=example)


# 3. Full Prompt Rendering ------------------------------------------------
def render_rate(stimulus_text: str, wording: str = "base") -> str:
    """Render the Rate prompt: instruction + blank line + stimulus."""
    instruction = _render_instruction("rate", wording, n=None)
    return f"{instruction}\n\n{stimulus_text.strip()}"


def render_bundle(task: str, stimulus_texts: list[str],
                  wording: str = "base") -> str:
    """Render Rank or Allocate: instruction + labeled-prose request list.

    Scaffold: single blank line between the instruction and the first
    request, single blank line between requests, no trailing separator.
    `Request N:` label on its own line above each stimulus.
    """
    if task not in ("rank", "allocate"):
        raise ValueError(f"render_bundle expects rank or allocate, got {task!r}")

    n = len(stimulus_texts)
    instruction = _render_instruction(task, wording, n=n)

    cfg = _load()
    label_tmpl = cfg["presentation"]["bundle"]["request_label"]

    blocks = [instruction]
    for i, text in enumerate(stimulus_texts, start=1):
        blocks.append(f"{label_tmpl.format(i=i)}\n{text.strip()}")
    return "\n\n".join(blocks)


def render_prompt(task: str, stimulus_texts: list[str],
                  wording: str = "base") -> str:
    """Dispatch on task: one renderer entry point for the instrument and
    the runner. `stimulus_texts` is an ordered list (length 1 for Rate)."""
    if task == "rate":
        if len(stimulus_texts) != 1:
            raise ValueError(f"rate expects 1 stimulus, got {len(stimulus_texts)}")
        return render_rate(stimulus_texts[0], wording)
    return render_bundle(task, stimulus_texts, wording)
