# FairFund-Bench — Codebook

Column reference for the three data files: `outcomes.csv` (the model
responses), `stimuli.csv` (the aid requests the models were shown), and
`models.csv` (the model lineup). All values are stored as human-readable labels
(e.g. race `White`/`Black`/`Hispanic`/`Asian`, gender `Male`/`Female`, framing
`no_cause`/`structural`/`self_cause`/`stigma_no_redemption`/`stigma_redemption`).

## `outcomes.csv` — 107,940 rows × 28 columns

One row per request the model evaluated. A Rate call is a single request and
contributes one row; each Rank or Allocate bundle is expanded to one row per
claimant, so the demographic and framing covariates for each claimant are found on
the same row as the model's response for that claimant.

The reported analyses use `wording_variant == "base" & valid`. The dataset
also includes three wording-variant collections (`need`, `merit`,
`paraphrase`) that were collected for robustness checks not analyzed in the
accompanying paper; they are found alongside `base` and can readily be filtered out.

| Column | Type | Description |
|---|---|---|
| `job_id`              | string  | API call identifier; one job = one model × one task × one bundle (or one Rate stimulus). |
| `model`               | string  | Model alias; key to `models.csv` (e.g. `claude-opus-4-6`). |
| `task`                | string  | `rate`, `rank`, or `allocate`. |
| `wording_variant`     | string  | Prompt variant: `base` (the main analyses) plus `need`, `merit`, `paraphrase` (robustness collections not analyzed in the accompanying paper). Filter to `base` to reproduce the paper. |
| `bundle_type`         | string  | Bundle type, decoded in the grid below. One of `rate`, `race_transparent`, `race_disguised`, `gender_transparent`, `gender_disguised`, `intersectional`, `framing_transparent`, `framing_disguised`. |
| `comparison_context`  | string  | `single` for Rate; `multi` for all bundle types. Matches the paper's single- vs. multi-stimulus distinction. |
| `mode`                | string  | Presentation mode within multi-stimulus prompts: `transparent` (focal axis varies, all else held fixed) or `disguised` (focal axis co-varies with scenario via a Latin/Graeco-Latin square). `NA` for Rate (mode is undefined outside multi-stimulus). |
| `focal_axis`          | string  | The within-bundle contrast the bundle identifies: `race`, `gender`, `intersectional`, or `framing`. `NA` for Rate (no within-bundle contrast). |
| `unit_id`           | string  | Rate: `stimulus_id`. Rank/Allocate: `bundle_id`. |
| `bundle_size`       | int     | Number of claimants in the prompt (1 for Rate; 2/4/5 for bundles). |
| `position`          | int     | 1-indexed position of the claimant within the prompt. Always 1 for Rate. |
| `stimulus_id`       | string  | The request shown at this position; key to `stimuli.csv`. |
| `category`          | string  | `Medical`, `Rent`, or `Education`. |
| `scenario_id`       | string  | Scenario template ID (e.g. `med_03`); 15 total (5 per category). |
| `name_rank`         | int     | 1–5 — which of the five validated name pairs in this race × gender cell was used. |
| `race`              | string  | Race signalled by the name: `White`, `Black`, `Hispanic`, `Asian`. |
| `gender`            | string  | Gender signalled by the name: `Male`, `Female`. |
| `framing`           | string  | Causal framing: `no_cause`, `structural`, `self_cause`, `stigma_no_redemption`, `stigma_redemption`. |
| `rating`            | int     | Rate answer (1–5). `NA` for Rank/Allocate rows and parse failures. |
| `rank`              | int     | Rank answer at this position (1 = highest priority). `NA` outside Rank. |
| `dollars`           | int     | Allocate answer at this position (dollars). `NA` outside Allocate. |
| `valid`             | logical | `TRUE` if parsed cleanly (not refused, not malformed, not missing). Use this to filter to analysis-ready rows. |
| `refused`           | logical | Model explicitly declined to answer (matched against a narrow set of refusal patterns). |
| `malformed`         | logical | Response present but not parseable to the task's required structure. |
| `safety_disclaimer` | logical | Response contains soft safety language (988 / SAMHSA / "please reach out to a professional" / etc.). Independent of `refused` — a response may carry a disclaimer alongside a valid structured answer. |
| `parse_reason`      | string  | Short diagnostic for refused/malformed rows; empty otherwise. |
| `retry_count`       | int     | Number of fallback API attempts used (0 = original parsed cleanly). |
| `missing`           | logical | `TRUE` if no raw response existed for this job. |

### Bundle-type grid

`bundle_type` is the single column that encodes the multi-stimulus design;
`comparison_context`, `mode`, and `focal_axis` are derivable from it and are
included for convenient filtering.

| `bundle_type`         | `comparison_context` | `mode`        | `focal_axis`     | Bundle size |
|-----------------------|----------------------|---------------|------------------|-------------|
| `rate`                | `single`             | `NA`          | `NA`             | 1 |
| `race_transparent`    | `multi`              | `transparent` | `race`           | 4 |
| `race_disguised`      | `multi`              | `disguised`   | `race`           | 4 |
| `gender_transparent`  | `multi`              | `transparent` | `gender`         | 2 |
| `gender_disguised`    | `multi`              | `disguised`   | `gender`         | 2 |
| `intersectional`      | `multi`              | `transparent` | `intersectional` | 4 |
| `framing_transparent` | `multi`              | `transparent` | `framing`        | 5 |
| `framing_disguised`   | `multi`              | `disguised`   | `framing`        | 5 |

Transparent bundles vary only the focal axis (a within-prompt minimal pair);
disguised bundles co-vary the focal axis with scenario so the contrast remains
identifiable across the pool but no single prompt is obviously an audit.

### Framing condition ladder

| `framing`              | Label             | Description |
|------------------------|----------------------------|-------------|
| `no_cause`             | No cause                   | No causal account given for the need. |
| `structural`           | Structural                 | Externally caused (e.g. layoff, illness). |
| `self_cause`           | Self-cause, mild          | Voluntary choice with mild blame (e.g. quit a job). |
| `stigma_no_redemption` | Stigma, no redemption      | High-blame cause (e.g. alcoholism) with no corrective action. |
| `stigma_redemption`    | Stigma, with redemption    | Same high-blame cause, paired with a stated corrective action (e.g. treatment). |

## `stimuli.csv` — 3,000 rows × 15 columns

Crosses 15 scenarios (5 per category) × 5 framings × 8 race × gender
cells × 5 name variants per cell. The full appeal text is included.

| Column | Type | Description |
|---|---|---|
| `stimulus_id`        | string  | Primary key. |
| `category`           | string  | `Medical`, `Rent`, `Education`. |
| `scenario_id`        | string  | `<cat>_NN`, e.g. `rent_03`. |
| `scenario_label`     | string  | Short human-readable scenario tag (e.g. `scholarship_alcohol`). |
| `framing`            | string  | See ladder above. |
| `race`               | string  | `White` / `Black` / `Hispanic` / `Asian`. |
| `gender`             | string  | `Male` / `Female`. |
| `name_rank`          | int     | Which of the 5 name pairs in this cell (1–5). |
| `first`, `last`      | string  | First and last name components ([Elder & Hayes 2023](https://www.journals.uchicago.edu/doi/abs/10.1086/723820) validated set). |
| `full_name`          | string  | The substituted name in the stimulus text. |
| `text`               | string  | Full first-person appeal text (opening + framing paragraph + closing). |
| `word_count`         | int     | Whitespace-tokenised word count of the rendered text. |
| `target_word_count`  | int     | Category-level target length, set from the median word count of real GoFundMe campaigns in that category (Medical 200, Rent 120, Education 190). |
| `within_tolerance`   | logical | `TRUE` if `word_count` is within ±25 of `target_word_count`. |

## `models.csv` — 14 rows × 5 columns

| Column | Type | Description |
|---|---|---|
| `model`         | string | Model alias (the key used in `outcomes.model`). |
| `display_name`  | string | Human-readable model name (e.g. `Opus 4.6`). |
| `provider`      | string | Anthropic / OpenAI / Google / xAI / Meta / DeepSeek / Mistral. |
| `tier`          | string | Capability tier: `Frontier`, `Mid`, `Mini`, or `Open-weight`. |
| `model_version` | string | Exact provider-side model version string(s) used at collection time. |
