"""
Program:   compose_bundles.py
Task:      Compose the FairFund-Bench bundle manifest (bundles.csv) from the
           stimulus universe (stimuli.csv) by the bundle-composition design
           of the accompanying paper (§3.3; Appendix G). This documents how
           the bundles are created and makes it possible to regenerate and
           extend them.

           Determinism. All randomness is controlled via numpy's default_rng
           using fixed per-task seeds, so re-running with the same seeds
           reproduces the shipped bundles.csv exactly (run with --check to
           assert this). Modifying the stimuli or seeds produces a new,
           valid instrument.

           Seven multi-stimulus bundle types per task plus the single-
           stimulus Rate manifest, each holding some factors fixed and
           varying a focal axis within the bundle via a Latin /
           Graeco-Latin square:

             bundle_type           size
             framing_disguised      5
             race_transparent       4
             race_disguised         4
             gender_transparent     2
             gender_disguised       2
             intersectional         4
             framing_transparent    5
             rate                   1

           Transparent types vary only the focal axis (a within-prompt
           minimal pair); disguised types co-vary it with scenario so no
           single prompt is obviously an audit.

           NOTE: the lowercase race / "control" framing tokens below are the
           construction-time codes embedded in every stimulus_id (e.g.
           `med_01__control__wf3`); they are how stimulus_ids are addressed
           and are unrelated to the human-readable labels in the data columns
           (`White`, `no_cause`, …). To adapt the benchmark to a different
           factor structure, edit these constants and the bundle builders.

Inputs:    benchmark/stimuli.csv   stimulus universe (uses category,
                                    scenario_id, stimulus_id)
Outputs:   benchmark/bundles.csv   task, unit_id, position, stimulus_id,
                                    bundle_type  (see benchmark/README.md)

Usage:     python benchmark/compose_bundles.py            # regenerate bundles.csv
           python benchmark/compose_bundles.py --check    # assert byte-identical, don't write

Project:   FairFund-Bench
Author:    Martin Lukk
"""

# 0. Program Setup --------------------------------------------------------
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent

CATEGORIES = ["Medical", "Rent", "Education"]
RACES      = ["white", "african_american", "hispanic", "asian"]
GENDERS    = ["F", "M"]
FRAMINGS   = ["control", "structural", "self_cause",
              "stigma_no_redemption", "stigma_redemption"]

RACE_CODE = {"white": "w", "african_american": "b",
             "hispanic": "h", "asian": "a"}
CAT_CODE  = {"Medical": "med", "Rent": "rnt", "Education": "edu"}

# Independent seeds per task. Re-running with these reproduces the shipped
# bundles.csv exactly.
SEEDS = {"rate": 20260418, "rank": 20260419, "allocate": 20260420}

# Framing-disguised Latin square shifts for (framing, scenario, name_rank).
# Each shift is coprime to 5 so each axis is a Latin square; distinct shifts
# form mutually orthogonal Latin squares, balancing pair-level counts within
# every bundle configuration.
FRAMING_DISGUISED_SHIFTS = (1, 2, 4)

# 4x4 Latin square for race-transparent bundles. Row = version, column =
# position; values index RACES.
RACE_TRANSPARENT_LATIN = np.array([
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
])

# Version pairs picking 2 of 4 versions per race-transparent configuration.
# Six pairs cycled across 30 configurations so each version is used 15 times.
RACE_TRANSPARENT_VERSION_PAIRS = [(0, 1), (0, 2), (0, 3),
                                  (1, 2), (1, 3), (2, 3)]

# Intersectional bundles: 2x2 (race x gender) over Black/White x M/F.
INTERSECTIONAL_CELLS = [("white", "F"), ("white", "M"),
                        ("african_american", "F"), ("african_american", "M")]
INTERSECTIONAL_LATIN = np.array([
    [0, 1, 2, 3],
    [1, 2, 3, 0],
    [2, 3, 0, 1],
    [3, 0, 1, 2],
])

# Race-disguised bundles: Graeco-Latin square of order 4. Each cell is
# (race_idx, scen_idx); every (race, scenario) pair appears once across the
# 16 cells, so race is orthogonal to scenario.
RACE_DISGUISED_GLS = [
    [(0, 0), (1, 1), (2, 2), (3, 3)],
    [(1, 2), (0, 3), (3, 0), (2, 1)],
    [(2, 3), (3, 2), (0, 1), (1, 0)],
    [(3, 1), (2, 0), (1, 3), (0, 2)],
]

# Gender-disguised bundles: size 2, one F + one M on two scenarios from a
# per-configuration pair. 4 versions cross gender flip x scenario flip.
GENDER_DISGUISED_VERSIONS = [
    [(1, "F", "a"), (2, "M", "b")],
    [(1, "M", "a"), (2, "F", "b")],
    [(1, "F", "b"), (2, "M", "a")],
    [(1, "M", "b"), (2, "F", "a")],
]
GENDER_DISGUISED_PAIRS = [(a, b) for a in range(5) for b in range(a + 1, 5)]


# 1. Helpers --------------------------------------------------------------
def stimulus_id(scenario_id: str, framing: str, race: str,
                gender: str, name_rank: int) -> str:
    return (f"{scenario_id}__{framing}__"
            f"{RACE_CODE[race]}{gender.lower()}{name_rank}")


def seeded_perm(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.permutation(n)


def build_scen_maps(stimuli: pd.DataFrame,
                    rng: np.random.Generator) -> dict:
    """Per-category framing->scenario map shared across the three transparent
    pools that hold framing fixed (race, gender, intersectional). Each framing
    gets a distinct scenario, held constant within every bundle of those
    types."""
    maps = {}
    for category in CATEGORIES:
        scen_ids = sorted(stimuli.loc[stimuli["category"] == category,
                                      "scenario_id"].unique())
        if len(scen_ids) != 5:
            raise ValueError(
                f"{category}: expected 5 scenarios, got {len(scen_ids)}"
            )
        maps[category] = dict(zip(
            FRAMINGS, [scen_ids[i] for i in seeded_perm(rng, 5)]
        ))
    return maps


# 2. Rate Stimulus Manifest -----------------------------------------------
def build_rate(stimuli: pd.DataFrame,
               rng: np.random.Generator) -> pd.DataFrame:
    """One name per (scenario x framing x race x gender): 15 x 5 x 8 = 600."""
    rows = []

    for category in CATEGORIES:
        scen_ids = sorted(stimuli.loc[stimuli["category"] == category,
                                      "scenario_id"].unique())
        for race in RACES:
            for gender in GENDERS:
                # Per-(race, gender, category) permutation of {1..5} maps the
                # 5 scenarios to the 5 name ranks in the cell's pool.
                rank_perm = [int(i) + 1 for i in seeded_perm(rng, 5)]
                for s_idx, scen_id in enumerate(scen_ids):
                    name_rank = rank_perm[s_idx]
                    for framing in FRAMINGS:
                        rows.append({
                            "category":    category,
                            "scenario_id": scen_id,
                            "framing":     framing,
                            "race":        race,
                            "gender":      gender,
                            "name_rank":   name_rank,
                            "stimulus_id": stimulus_id(
                                scen_id, framing, race, gender, name_rank
                            ),
                        })
    return pd.DataFrame(rows)


# 3. Framing-Disguised Bundles --------------------------------------------
def build_framing_disguised(stimuli: pd.DataFrame,
                            rng: np.random.Generator) -> pd.DataFrame:
    shift_f, shift_s, shift_n = FRAMING_DISGUISED_SHIFTS
    rows = []

    for category in CATEGORIES:
        scen_ids = sorted(stimuli.loc[stimuli["category"] == category,
                                      "scenario_id"].unique())
        if len(scen_ids) != 5:
            raise ValueError(f"{category}: expected 5 scenarios, got {len(scen_ids)}")

        for race in RACES:
            for gender in GENDERS:
                # Per-cell shuffled orderings; Latin-square shifts operate on
                # these permuted axes, giving cell-level independence.
                f_order = [FRAMINGS[i] for i in seeded_perm(rng, 5)]
                s_order = [scen_ids[i] for i in seeded_perm(rng, 5)]
                n_order = [int(i) + 1 for i in seeded_perm(rng, 5)]

                for v in range(5):
                    bundle_id = (f"prim__{CAT_CODE[category]}_"
                                 f"{RACE_CODE[race]}{gender.lower()}_v{v + 1}")
                    for p in range(5):
                        f_idx = (shift_f * v + p) % 5
                        s_idx = (shift_s * v + p) % 5
                        n_idx = (shift_n * v + p) % 5

                        framing   = f_order[f_idx]
                        scen_id   = s_order[s_idx]
                        name_rank = n_order[n_idx]

                        rows.append({
                            "bundle_id":   bundle_id,
                            "bundle_type": "framing_disguised",
                            "version":     v + 1,
                            "position":    p + 1,
                            "bundle_size": 5,
                            "category":    category,
                            "race":        race,
                            "gender":      gender,
                            "framing":     framing,
                            "scenario_id": scen_id,
                            "name_rank":   name_rank,
                            "stimulus_id": stimulus_id(
                                scen_id, framing, race, gender, name_rank
                            ),
                        })
    return pd.DataFrame(rows)


# 4. Race-Transparent Bundles ---------------------------------------------
def build_race_transparent(stimuli: pd.DataFrame,
                           rng: np.random.Generator,
                           scen_maps: dict) -> pd.DataFrame:
    rows = []
    config_idx = 0

    # Name-rank rotation per (race, gender): 60 draws per cell over 5 ranks.
    name_perms    = {(r, g): [int(i) + 1 for i in seeded_perm(rng, 5)]
                     for r in RACES for g in GENDERS}
    name_counters = {(r, g): 0 for r in RACES for g in GENDERS}

    for category in CATEGORIES:
        framing_to_scen = scen_maps[category]

        for gender in GENDERS:
            for framing in FRAMINGS:
                v_pair  = RACE_TRANSPARENT_VERSION_PAIRS[
                    config_idx % len(RACE_TRANSPARENT_VERSION_PAIRS)
                ]
                scen_id = framing_to_scen[framing]

                for version in v_pair:
                    bundle_id = (f"comp__{CAT_CODE[category]}_{gender.lower()}_"
                                 f"{framing}_v{version + 1}")
                    for p in range(4):
                        race = RACES[RACE_TRANSPARENT_LATIN[version, p]]
                        key  = (race, gender)
                        name_rank = name_perms[key][name_counters[key] % 5]
                        name_counters[key] += 1

                        rows.append({
                            "bundle_id":   bundle_id,
                            "bundle_type": "race_transparent",
                            "version":     version + 1,
                            "position":    p + 1,
                            "bundle_size": 4,
                            "category":    category,
                            "race":        race,
                            "gender":      gender,
                            "framing":     framing,
                            "scenario_id": scen_id,
                            "name_rank":   name_rank,
                            "stimulus_id": stimulus_id(
                                scen_id, framing, race, gender, name_rank
                            ),
                        })
                config_idx += 1

    return pd.DataFrame(rows)


# 5. Gender-Transparent Bundles -------------------------------------------
def build_gender_transparent(stimuli: pd.DataFrame,
                             rng: np.random.Generator,
                             scen_maps: dict) -> pd.DataFrame:
    """Size-2 minimal pairs (F vs. M) holding category, race, framing, and
    scenario constant. 60 configurations x 2 versions; v1 = F-first,
    v2 = M-first, fully balancing gender x position."""
    rows = []

    name_perms    = {(r, g): [int(i) + 1 for i in seeded_perm(rng, 5)]
                     for r in RACES for g in GENDERS}
    name_counters = {(r, g): 0 for r in RACES for g in GENDERS}

    VERSION_ORDERS = [
        [(1, "F"), (2, "M")],
        [(1, "M"), (2, "F")],
    ]

    for category in CATEGORIES:
        framing_to_scen = scen_maps[category]
        for race in RACES:
            for framing in FRAMINGS:
                scen_id = framing_to_scen[framing]
                for v_idx, order in enumerate(VERSION_ORDERS):
                    bundle_id = (f"gcomp__{CAT_CODE[category]}_{RACE_CODE[race]}_"
                                 f"{framing}_v{v_idx + 1}")
                    for position, gender in order:
                        key = (race, gender)
                        name_rank = name_perms[key][name_counters[key] % 5]
                        name_counters[key] += 1

                        rows.append({
                            "bundle_id":   bundle_id,
                            "bundle_type": "gender_transparent",
                            "version":     v_idx + 1,
                            "position":    position,
                            "bundle_size": 2,
                            "category":    category,
                            "race":        race,
                            "gender":      gender,
                            "framing":     framing,
                            "scenario_id": scen_id,
                            "name_rank":   name_rank,
                            "stimulus_id": stimulus_id(
                                scen_id, framing, race, gender, name_rank
                            ),
                        })

    return pd.DataFrame(rows)


# 6. Intersectional Bundles -----------------------------------------------
def build_intersectional(stimuli: pd.DataFrame,
                         rng: np.random.Generator,
                         scen_maps: dict) -> pd.DataFrame:
    """Size-4 bundles over the Black/White x M/F subfactorial. 15
    configurations (category x framing) x 4 versions, a full 4x4 Latin square
    balancing each (race x gender) cell across all 4 positions."""
    rows = []

    cells = INTERSECTIONAL_CELLS
    name_perms    = {c: [int(i) + 1 for i in seeded_perm(rng, 5)] for c in cells}
    name_counters = {c: 0 for c in cells}

    for category in CATEGORIES:
        framing_to_scen = scen_maps[category]
        for framing in FRAMINGS:
            scen_id = framing_to_scen[framing]
            for version in range(4):
                bundle_id = (f"inter__{CAT_CODE[category]}_{framing}_"
                             f"v{version + 1}")
                for p in range(4):
                    race, gender = cells[INTERSECTIONAL_LATIN[version, p]]
                    key = (race, gender)
                    name_rank = name_perms[key][name_counters[key] % 5]
                    name_counters[key] += 1

                    rows.append({
                        "bundle_id":   bundle_id,
                        "bundle_type": "intersectional",
                        "version":     version + 1,
                        "position":    p + 1,
                        "bundle_size": 4,
                        "category":    category,
                        "race":        race,
                        "gender":      gender,
                        "framing":     framing,
                        "scenario_id": scen_id,
                        "name_rank":   name_rank,
                        "stimulus_id": stimulus_id(
                            scen_id, framing, race, gender, name_rank
                        ),
                    })

    return pd.DataFrame(rows)


# 7. Race-Disguised Bundles -----------------------------------------------
def build_race_disguised(stimuli: pd.DataFrame,
                         rng: np.random.Generator) -> pd.DataFrame:
    """Race co-varied with scenario. Holds (category x gender x framing) and
    varies race + scenario jointly within bundle (size 4: 4 races, 4 of 5
    scenarios, one sits out per bundle). A Graeco-Latin square of order 4
    balances race x position and scenario x position and pairs every (race,
    scenario) once. 30 configurations x 4 versions = 480 rows/task."""
    rows = []

    name_perms    = {(r, g): [int(i) + 1 for i in seeded_perm(rng, 5)]
                     for r in RACES for g in GENDERS}
    name_counters = {(r, g): 0 for r in RACES for g in GENDERS}

    cat_scen_orders = {}
    cat_race_orders = {}
    for category in CATEGORIES:
        scen_ids = sorted(stimuli.loc[stimuli["category"] == category,
                                      "scenario_id"].unique())
        if len(scen_ids) != 5:
            raise ValueError(f"{category}: expected 5 scenarios, got {len(scen_ids)}")
        cat_scen_orders[category] = [scen_ids[i] for i in seeded_perm(rng, 5)]
        cat_race_orders[category] = [RACES[i]    for i in seeded_perm(rng, 4)]

    config_idx = 0
    for category in CATEGORIES:
        scen_order = cat_scen_orders[category]
        race_order = cat_race_orders[category]
        for gender in GENDERS:
            for framing in FRAMINGS:
                skip_idx = config_idx % 5
                used = [scen_order[i] for i in range(5) if i != skip_idx]

                for v in range(4):
                    bundle_id = (f"compv__{CAT_CODE[category]}_{gender.lower()}_"
                                 f"{framing}_v{v + 1}")
                    for p in range(4):
                        race_idx, scen_idx = RACE_DISGUISED_GLS[v][p]
                        race    = race_order[race_idx]
                        scen_id = used[scen_idx]

                        key = (race, gender)
                        name_rank = name_perms[key][name_counters[key] % 5]
                        name_counters[key] += 1

                        rows.append({
                            "bundle_id":   bundle_id,
                            "bundle_type": "race_disguised",
                            "version":     v + 1,
                            "position":    p + 1,
                            "bundle_size": 4,
                            "category":    category,
                            "race":        race,
                            "gender":      gender,
                            "framing":     framing,
                            "scenario_id": scen_id,
                            "name_rank":   name_rank,
                            "stimulus_id": stimulus_id(
                                scen_id, framing, race, gender, name_rank
                            ),
                        })
                config_idx += 1

    return pd.DataFrame(rows)


# 8. Gender-Disguised Bundles ---------------------------------------------
def build_gender_disguised(stimuli: pd.DataFrame,
                           rng: np.random.Generator) -> pd.DataFrame:
    """Gender co-varied with scenario. Holds (category x race x framing) and
    varies gender + scenario jointly within bundle (size 2: one F + one M on
    two different scenarios). The disguised parallel to the gender-transparent
    bundles. 60 configurations x 4 versions = 480 rows/task; the 4 versions
    cross gender flip x scenario flip, balancing (gender x scenario x
    position)."""
    rows = []

    name_perms    = {(r, g): [int(i) + 1 for i in seeded_perm(rng, 5)]
                     for r in RACES for g in GENDERS}
    name_counters = {(r, g): 0 for r in RACES for g in GENDERS}

    cat_scen_orders = {}
    for category in CATEGORIES:
        scen_ids = sorted(stimuli.loc[stimuli["category"] == category,
                                      "scenario_id"].unique())
        if len(scen_ids) != 5:
            raise ValueError(f"{category}: expected 5 scenarios, got {len(scen_ids)}")
        cat_scen_orders[category] = [scen_ids[i] for i in seeded_perm(rng, 5)]

    config_idx = 0
    for category in CATEGORIES:
        scen_order = cat_scen_orders[category]
        for race in RACES:
            for framing in FRAMINGS:
                pair_idx = config_idx % len(GENDER_DISGUISED_PAIRS)
                a_idx, b_idx = GENDER_DISGUISED_PAIRS[pair_idx]
                sc_a = scen_order[a_idx]
                sc_b = scen_order[b_idx]
                scen_lookup = {"a": sc_a, "b": sc_b}

                for v_idx, version in enumerate(GENDER_DISGUISED_VERSIONS):
                    bundle_id = (f"gcompv__{CAT_CODE[category]}_{RACE_CODE[race]}_"
                                 f"{framing}_v{v_idx + 1}")
                    for position, gender, scen_tag in version:
                        scen_id = scen_lookup[scen_tag]
                        key = (race, gender)
                        name_rank = name_perms[key][name_counters[key] % 5]
                        name_counters[key] += 1

                        rows.append({
                            "bundle_id":   bundle_id,
                            "bundle_type": "gender_disguised",
                            "version":     v_idx + 1,
                            "position":    position,
                            "bundle_size": 2,
                            "category":    category,
                            "race":        race,
                            "gender":      gender,
                            "framing":     framing,
                            "scenario_id": scen_id,
                            "name_rank":   name_rank,
                            "stimulus_id": stimulus_id(
                                scen_id, framing, race, gender, name_rank
                            ),
                        })
                config_idx += 1

    return pd.DataFrame(rows)


# 9. Framing-Transparent Bundles ------------------------------------------
def build_framing_transparent(stimuli: pd.DataFrame,
                              rng: np.random.Generator) -> pd.DataFrame:
    """Size-5 minimal-pair pool varying framing within bundle (the framing-
    axis analog of the race- and gender-transparent bundles). Holds (category
    x race x gender x scenario) constant; each of the 5 framings appears once
    at one of 5 positions. 120 cells x 1 version = 600 rows. Each cell uses
    one row of a 5-row cyclic Latin square; rows are cycled across cells via a
    seeded permutation so each row id is hit 24 times (exact pool-level
    framing x position balance)."""
    rows = []

    cat_scen = {c: sorted(stimuli.loc[stimuli["category"] == c, "scenario_id"].unique())
                for c in CATEGORIES}
    for c in CATEGORIES:
        if len(cat_scen[c]) != 5:
            raise ValueError(f"{c}: expected 5 scenarios, got {len(cat_scen[c])}")

    cells = [(category, race, gender, s_idx)
             for category in CATEGORIES
             for race in RACES
             for gender in GENDERS
             for s_idx in range(5)]
    row_perm = seeded_perm(rng, len(cells))

    for cell_idx, (category, race, gender, s_idx) in enumerate(cells):
        scen_id = cat_scen[category][s_idx]
        n_order = [int(i) + 1 for i in seeded_perm(rng, 5)]
        latin_row = int(row_perm[cell_idx]) % 5

        bundle_id = (f"framcomp__{RACE_CODE[race]}{gender.lower()}_"
                     f"{scen_id}_v1")

        for p in range(5):
            framing   = FRAMINGS[(latin_row + p) % 5]
            name_rank = n_order[p]

            rows.append({
                "bundle_id":   bundle_id,
                "bundle_type": "framing_transparent",
                "version":     1,
                "position":    p + 1,
                "bundle_size": 5,
                "category":    category,
                "race":        race,
                "gender":      gender,
                "framing":     framing,
                "scenario_id": scen_id,
                "name_rank":   name_rank,
                "stimulus_id": stimulus_id(
                    scen_id, framing, race, gender, name_rank
                ),
            })

    return pd.DataFrame(rows)


# 10. Design Validation ---------------------------------------------------
def validate(bundles: pd.DataFrame, stimuli: pd.DataFrame, task: str) -> None:
    valid = set(stimuli["stimulus_id"])
    missing = set(bundles["stimulus_id"]) - valid
    if missing:
        raise ValueError(
            f"{task}: {len(missing)} stimulus_ids not in stimuli.csv "
            f"(first 5: {sorted(missing)[:5]})"
        )

    fram_dis = bundles[bundles["bundle_type"] == "framing_disguised"]
    if fram_dis["bundle_id"].nunique() != 120 or len(fram_dis) != 600:
        raise ValueError(
            f"{task} framing_disguised: {fram_dis['bundle_id'].nunique()} "
            f"bundles, {len(fram_dis)} rows (expected 120 / 600)"
        )
    fp = fram_dis.groupby(["framing", "position"]).size().unstack(fill_value=0)
    if not (fp.values == 24).all():
        raise ValueError(
            f"{task}: framing×position imbalance in framing_disguised:\n{fp}"
        )
    within = fram_dis.groupby("bundle_id")["framing"].nunique()
    if not (within == 5).all():
        raise ValueError(
            f"{task}: framing_disguised bundles without 5 distinct framings"
        )

    race_tr = bundles[bundles["bundle_type"] == "race_transparent"]
    if race_tr["bundle_id"].nunique() != 60 or len(race_tr) != 240:
        raise ValueError(
            f"{task} race_transparent: {race_tr['bundle_id'].nunique()} "
            f"bundles, {len(race_tr)} rows (expected 60 / 240)"
        )
    rp = race_tr.groupby(["race", "position"]).size().unstack(fill_value=0)
    if not (rp.values == 15).all():
        raise ValueError(
            f"{task}: race×position imbalance in race_transparent:\n{rp}"
        )
    within_race = race_tr.groupby("bundle_id")["race"].nunique()
    if not (within_race == 4).all():
        raise ValueError(
            f"{task}: race_transparent bundles without 4 distinct races"
        )

    gender_tr = bundles[bundles["bundle_type"] == "gender_transparent"]
    if gender_tr["bundle_id"].nunique() != 120 or len(gender_tr) != 240:
        raise ValueError(
            f"{task} gender_transparent: {gender_tr['bundle_id'].nunique()} "
            f"bundles, {len(gender_tr)} rows (expected 120 / 240)"
        )
    gp = gender_tr.groupby(["gender", "position"]).size().unstack(fill_value=0)
    if not (gp.values == 60).all():
        raise ValueError(
            f"{task}: gender×position imbalance in gender_transparent:\n{gp}"
        )
    within_gt = gender_tr.groupby("bundle_id").agg({
        "category": "nunique", "race": "nunique", "framing": "nunique",
        "scenario_id": "nunique", "gender": "nunique",
    })
    if not ((within_gt[["category", "race", "framing", "scenario_id"]] == 1).all().all()
            and (within_gt["gender"] == 2).all()):
        raise ValueError(
            f"{task}: gender_transparent bundles fail minimal-pair integrity"
        )

    inter = bundles[bundles["bundle_type"] == "intersectional"]
    if inter["bundle_id"].nunique() != 60 or len(inter) != 240:
        raise ValueError(
            f"{task} intersectional: {inter['bundle_id'].nunique()} "
            f"bundles, {len(inter)} rows (expected 60 / 240)"
        )
    cell = inter["race"].astype(str) + "_" + inter["gender"].astype(str)
    rgp  = inter.groupby([cell, inter["position"]]).size().unstack(fill_value=0)
    if not (rgp.values == 15).all():
        raise ValueError(
            f"{task}: (race × gender) × position imbalance in intersectional:\n{rgp}"
        )
    within_inter = inter.groupby("bundle_id").apply(
        lambda df: df[["race", "gender"]].drop_duplicates().shape[0]
    )
    if not (within_inter == 4).all():
        raise ValueError(
            f"{task}: intersectional bundles without 4 distinct (race, gender) cells"
        )

    race_dis = bundles[bundles["bundle_type"] == "race_disguised"]
    if race_dis["bundle_id"].nunique() != 120 or len(race_dis) != 480:
        raise ValueError(
            f"{task} race_disguised: {race_dis['bundle_id'].nunique()} "
            f"bundles, {len(race_dis)} rows (expected 120 / 480)"
        )
    rp = race_dis.groupby(["race", "position"]).size().unstack(fill_value=0)
    if not (rp.values == 30).all():
        raise ValueError(
            f"{task}: race×position imbalance in race_disguised:\n{rp}"
        )
    sp = race_dis.groupby(["scenario_id", "position"]).size().unstack(fill_value=0)
    if not (sp.values == 8).all():
        raise ValueError(
            f"{task}: scenario×position imbalance in race_disguised:\n{sp}"
        )
    within_rd = race_dis.groupby("bundle_id").agg({
        "race": "nunique", "scenario_id": "nunique",
        "category": "nunique", "gender": "nunique", "framing": "nunique",
    })
    if not (((within_rd[["race", "scenario_id"]] == 4).all().all())
            and ((within_rd[["category", "gender", "framing"]] == 1).all().all())):
        raise ValueError(
            f"{task}: race_disguised bundles fail (race × scenario) integrity"
        )

    gender_dis = bundles[bundles["bundle_type"] == "gender_disguised"]
    if gender_dis["bundle_id"].nunique() != 240 or len(gender_dis) != 480:
        raise ValueError(
            f"{task} gender_disguised: {gender_dis['bundle_id'].nunique()} "
            f"bundles, {len(gender_dis)} rows (expected 240 / 480)"
        )
    gp = gender_dis.groupby(["gender", "position"]).size().unstack(fill_value=0)
    if not (gp.values == 120).all():
        raise ValueError(
            f"{task}: gender×position imbalance in gender_disguised:\n{gp}"
        )
    sp = gender_dis.groupby(["scenario_id", "position"]).size().unstack(fill_value=0)
    if not (sp.values == 16).all():
        raise ValueError(
            f"{task}: scenario×position imbalance in gender_disguised:\n{sp}"
        )
    within_gd = gender_dis.groupby("bundle_id").agg({
        "gender": "nunique", "scenario_id": "nunique",
        "category": "nunique", "race": "nunique", "framing": "nunique",
    })
    if not (((within_gd[["gender", "scenario_id"]] == 2).all().all())
            and ((within_gd[["category", "race", "framing"]] == 1).all().all())):
        raise ValueError(
            f"{task}: gender_disguised bundles fail (gender × scenario) integrity"
        )

    fram_tr = bundles[bundles["bundle_type"] == "framing_transparent"]
    if fram_tr["bundle_id"].nunique() != 120 or len(fram_tr) != 600:
        raise ValueError(
            f"{task} framing_transparent: {fram_tr['bundle_id'].nunique()} "
            f"bundles, {len(fram_tr)} rows (expected 120 / 600)"
        )
    fp = fram_tr.groupby(["framing", "position"]).size().unstack(fill_value=0)
    if not (fp.values == 24).all():
        raise ValueError(
            f"{task}: framing×position imbalance in framing_transparent:\n{fp}"
        )
    within_ft = fram_tr.groupby("bundle_id").agg({
        "framing": "nunique", "category": "nunique", "race": "nunique",
        "gender": "nunique", "scenario_id": "nunique",
    })
    if not (((within_ft["framing"] == 5).all())
            and ((within_ft[["category", "race", "gender", "scenario_id"]] == 1)
                 .all().all())):
        raise ValueError(
            f"{task}: framing_transparent bundles fail within-bundle integrity"
        )


def validate_rate(rate: pd.DataFrame, stimuli: pd.DataFrame) -> None:
    valid = set(stimuli["stimulus_id"])
    missing = set(rate["stimulus_id"]) - valid
    if missing:
        raise ValueError(f"rate: {len(missing)} stimulus_ids not in stimuli.csv")
    if len(rate) != 600:
        raise ValueError(f"rate: expected 600 stimuli, got {len(rate)}")
    if rate["stimulus_id"].duplicated().any():
        raise ValueError("rate: duplicate stimulus_ids")
    cell_counts = rate.groupby(["category", "race", "gender", "framing"]).size()
    if not (cell_counts == 5).all():
        raise ValueError(f"rate: uneven per-cell N (expected 5):\n{cell_counts}")


# 11. Compose + Emit ------------------------------------------------------
def compose(stimuli: pd.DataFrame) -> pd.DataFrame:
    """Build all pools and return the release bundles.csv frame."""
    rate = build_rate(stimuli, np.random.default_rng(SEEDS["rate"]))
    validate_rate(rate, stimuli)

    task_bundles = {}
    for task in ("rank", "allocate"):
        rng = np.random.default_rng(SEEDS[task])
        fram_dis   = build_framing_disguised(stimuli, rng)
        scen_maps  = build_scen_maps(stimuli, rng)
        race_tr    = build_race_transparent(stimuli, rng, scen_maps)
        gender_tr  = build_gender_transparent(stimuli, rng, scen_maps)
        inter      = build_intersectional(stimuli, rng, scen_maps)
        # Appended last so the earlier pools' rng draws stay byte-identical.
        race_dis   = build_race_disguised(stimuli, rng)
        gender_dis = build_gender_disguised(stimuli, rng)
        fram_tr    = build_framing_transparent(stimuli, rng)
        bundles = pd.concat(
            [fram_dis, race_tr, gender_tr, inter,
             race_dis, gender_dis, fram_tr],
            ignore_index=True,
        )
        validate(bundles, stimuli, task)
        task_bundles[task] = bundles

    # Project the design frame onto the release schema.
    def to_release(b: pd.DataFrame, task: str) -> pd.DataFrame:
        return pd.DataFrame({
            "task":        task,
            "unit_id":     b["bundle_id"],
            "position":    b["position"],
            "stimulus_id": b["stimulus_id"],
            "bundle_type": b["bundle_type"],
        })

    rate_rel = pd.DataFrame({
        "task": "rate", "unit_id": rate["stimulus_id"], "position": 1,
        "stimulus_id": rate["stimulus_id"], "bundle_type": "rate",
    })
    out = pd.concat(
        [rate_rel, to_release(task_bundles["rank"], "rank"),
         to_release(task_bundles["allocate"], "allocate")],
        ignore_index=True,
    ).sort_values(["task", "unit_id", "position"]).reset_index(drop=True)

    return out


# 12. Main ----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stimuli", type=Path, default=SCRIPT_DIR / "stimuli.csv")
    ap.add_argument("--out",     type=Path, default=SCRIPT_DIR / "bundles.csv")
    ap.add_argument("--check", action="store_true",
                    help="assert the composed bundles match an existing --out "
                         "exactly; do not write")
    args = ap.parse_args()

    stimuli = pd.read_csv(args.stimuli, low_memory=False)
    print(f"Loaded {len(stimuli):,} stimuli from {args.stimuli.name}.")
    print(f"Seeds: rate={SEEDS['rate']}, rank={SEEDS['rank']}, "
          f"allocate={SEEDS['allocate']}.")

    out = compose(stimuli)
    n_units = out[["task", "unit_id"]].drop_duplicates().shape[0]
    print(f"Composed {len(out):,} rows / {n_units:,} units.")

    if args.check:
        if not args.out.exists():
            raise SystemExit(f"--check: {args.out} does not exist to compare against")
        existing = pd.read_csv(args.out, low_memory=False)
        identical = out.equals(existing[out.columns])
        print(f"byte-identical to {args.out.name}: {identical}")
        if not identical:
            raise SystemExit("composed bundles DIFFER from existing bundles.csv")
        return

    out.to_csv(args.out, index=False)
    print(f"-> {args.out}")
    print("\nUnits by bundle_type:")
    print(out.drop_duplicates(["task", "unit_id"])
             .groupby("bundle_type").size().to_string())


if __name__ == "__main__":
    main()
