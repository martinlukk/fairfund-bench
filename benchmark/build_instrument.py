"""
Program:   build_instrument.py
Task:      Materialise the FairFund-Bench instrument: the fixed set of
           prompts the benchmark presents, independent of any model. For
           each (task, unit_id) it emits the ordered claimant list, the
           bundle type, and the fully rendered prompt string a runner would
           send. This is the hand-off between the discrete instrument inputs
           and the Inspect runner (benchmark/fairfund.py).

           The instrument is a join of three discrete inputs, all local
           to benchmark/:

             benchmark/bundles.csv   which stimulus sits at which position in
                                     which bundle (the composition)
             benchmark/stimuli.csv   the appeal text + per-claimant covariates
             benchmark/prompts.yaml  the locked prompt wordings (via render.py)

           data/outcomes.csv is the benchmark's OUTPUT, not an input here. To
           adapt the benchmark to a new domain, edit stimuli.csv (new appeal
           texts) and bundles.csv (how they are composed) and re-run this
           script — see benchmark/README.md. (benchmark/stimuli.csv is the
           instrument copy of the released data/stimuli.csv; they are identical.)

           Only the base wording is materialised — that is the benchmark
           reported in the paper. The need / merit / paraphrase robustness
           wordings are not part of the instrument.

Inputs:    benchmark/bundles.csv   bundle composition (task, unit_id, position,
                                    stimulus_id, bundle_type)
           benchmark/stimuli.csv   appeal text + per-claimant covariates
           benchmark/prompts.yaml  locked prompt wordings (via render.py)
Outputs:   benchmark/instrument.csv   one row per (task, unit_id): structure
                                       + ordered stimulus_ids + rendered prompt
           benchmark/instrument.json  same, with per-position covariates
                                       nested (the runner's Sample.metadata)

Usage:     python benchmark/build_instrument.py

Project:   FairFund-Bench
Author:    Martin Lukk
"""

# 0. Program Setup --------------------------------------------------------
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from render import render_prompt

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT       = SCRIPT_DIR.parent

# Decode bundle_type into the design coordinates the codebook lists as
# derivable from it. comparison_context / mode / focal_axis for each of the
# eight released bundle types. A new bundle_type added for a domain
# adaptation needs one entry here (see README).
BUNDLE_TYPE_GRID = {
    "rate":                ("single", "NA",          "NA"),
    "race_transparent":    ("multi",  "transparent", "race"),
    "race_disguised":      ("multi",  "disguised",   "race"),
    "gender_transparent":  ("multi",  "transparent", "gender"),
    "gender_disguised":    ("multi",  "disguised",   "gender"),
    "intersectional":      ("multi",  "transparent", "intersectional"),
    "framing_transparent": ("multi",  "transparent", "framing"),
    "framing_disguised":   ("multi",  "disguised",   "framing"),
}

# Position-level covariates carried into Sample.metadata, sourced from the
# authoritative stimulus table (keyed on stimulus_id).
POSITION_COVARIATES = ("category", "scenario_id", "scenario_label",
                       "framing", "race", "gender", "name_rank",
                       "first", "last", "full_name")


# 1. Load Discrete Inputs -------------------------------------------------
def load_inputs(bundles_path: Path, stimuli_path: Path):
    bundles = pd.read_csv(bundles_path, low_memory=False)
    stimuli = pd.read_csv(stimuli_path, low_memory=False)

    missing_text = set(bundles["stimulus_id"]) - set(stimuli["stimulus_id"])
    if missing_text:
        raise SystemExit(
            f"{len(missing_text)} stimulus_id(s) in bundles.csv have no text in "
            f"stimuli.csv, e.g. {sorted(missing_text)[:3]}")

    unknown = set(bundles["bundle_type"]) - set(BUNDLE_TYPE_GRID)
    if unknown:
        raise SystemExit(
            f"bundle_type(s) {sorted(unknown)} not in BUNDLE_TYPE_GRID; add an "
            f"entry decoding comparison_context / mode / focal_axis.")

    return bundles, stimuli.set_index("stimulus_id")


# 2. Assemble Units -------------------------------------------------------
def build_units(bundles: pd.DataFrame, stimuli_ix: pd.DataFrame, wording: str):
    """One record per (task, unit_id): ordered positions + rendered prompt."""
    records = []
    for (task, unit_id), grp in bundles.groupby(["task", "unit_id"], sort=True):
        grp = grp.sort_values("position")

        positions = grp["position"].tolist()
        n = len(positions)
        if positions != list(range(1, n + 1)):
            raise SystemExit(f"{task}/{unit_id}: non-contiguous positions {positions}")
        if grp["stimulus_id"].duplicated().any():
            raise SystemExit(f"{task}/{unit_id}: duplicate stimulus within bundle")
        bundle_types = grp["bundle_type"].unique()
        if len(bundle_types) != 1:
            raise SystemExit(f"{task}/{unit_id}: bundle_type not constant ({bundle_types})")
        bundle_type = str(bundle_types[0])
        comparison_context, mode, focal_axis = BUNDLE_TYPE_GRID[bundle_type]

        ordered_sids = grp["stimulus_id"].tolist()
        texts        = [stimuli_ix.at[sid, "text"] for sid in ordered_sids]
        prompt       = render_prompt(task, texts, wording=wording)

        position_meta = []
        for pos, sid in zip(positions, ordered_sids):
            row = stimuli_ix.loc[sid]
            meta = {"position": int(pos), "stimulus_id": sid}
            for cov in POSITION_COVARIATES:
                val = row[cov]
                meta[cov] = int(val) if cov == "name_rank" else val
            position_meta.append(meta)

        records.append({
            "task": task,
            "unit_id": unit_id,
            "wording_variant": wording,
            "bundle_type": bundle_type,
            "bundle_size": n,
            "comparison_context": comparison_context,
            "mode": mode,
            "focal_axis": focal_axis,
            "stimulus_ids": ordered_sids,
            "prompt": prompt,
            "positions": position_meta,
        })

    records.sort(key=lambda r: (r["task"], r["unit_id"]))
    return records


# 3. Emit -----------------------------------------------------------------
def write_outputs(records, csv_path: Path, json_path: Path):
    with open(json_path, "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)

    flat = pd.DataFrame([{
        "task": r["task"],
        "unit_id": r["unit_id"],
        "wording_variant": r["wording_variant"],
        "bundle_type": r["bundle_type"],
        "bundle_size": r["bundle_size"],
        "comparison_context": r["comparison_context"],
        "mode": r["mode"],
        "focal_axis": r["focal_axis"],
        "stimulus_ids": "|".join(r["stimulus_ids"]),
        "prompt": r["prompt"],
    } for r in records])
    flat.to_csv(csv_path, index=False)
    return flat


# 4. Main -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundles",  type=Path, default=SCRIPT_DIR / "bundles.csv")
    ap.add_argument("--stimuli",  type=Path, default=SCRIPT_DIR / "stimuli.csv")
    ap.add_argument("--wording",  default="base",
                    help="wording variant to materialise (default: base)")
    ap.add_argument("--out-csv",  type=Path, default=SCRIPT_DIR / "instrument.csv")
    ap.add_argument("--out-json", type=Path, default=SCRIPT_DIR / "instrument.json")
    args = ap.parse_args()

    bundles, stimuli_ix = load_inputs(args.bundles, args.stimuli)
    print(f"Loaded {len(bundles):,} composition rows; "
          f"{bundles[['task','unit_id']].drop_duplicates().shape[0]:,} units.")

    records = build_units(bundles, stimuli_ix, args.wording)
    flat = write_outputs(records, args.out_csv, args.out_json)

    print(f"\nMaterialised {len(records):,} units -> {args.out_csv.name}, "
          f"{args.out_json.name}")
    print("\nUnits by task:")
    print(flat.groupby("task").size().to_string())
    print("\nUnits by bundle_type:")
    print(flat.groupby("bundle_type").size().to_string())


if __name__ == "__main__":
    main()
