# FairFund-Bench — Model Output Dataset

Companion data for *FairFund-Bench: Evaluating Deservingness Bias in LLM
Allocation Decisions*: parsed responses from 14 LLMs to
the benchmark, together with the stimuli they were shown and the model lineup.

## Files

| File | Rows | Description |
|---|---|---|
| `outcomes.csv` | 107,940 | One row per evaluated request — a Rate call is a single row, while each Rank/Allocate bundle is expanded to one row per claimant, containing values for that claimant's covariates. |
| `stimuli.csv`  | 3,000   | The stimulus universe, with the full first-person appeal text for each request. |
| `models.csv`   | 14      | Model lineup (alias, provider, tier, provider-side version). |
| `codebook.md`  | —       | Column-by-column documentation of all three tables, including the `bundle_type` design grid. |

`outcomes.csv` is the main table: `stimulus_id` joins it to `stimuli.csv` and
`model` joins it to `models.csv`. See `codebook.md` for the full column
reference and grid with `bundle_type` → `mode` → `focal_axis`.

## Experiments

Each of the 14 LLMs evaluated 600 single-stimulus prompts (the Rate task) and
840 multi-stimulus bundles per task (Rank and Allocate), crossing:

- **Three categories** of financial need — Medical, Rent, Education.
- **Five causal framings** of need — `no_cause`, `structural`, `self_cause`, `stigma_no_redemption`, `stigma_redemption` (from van Oorschot's *Control* dimension; [van Oorschot & Roosma 2017](https://doi.org/10.4337/9781785367212.00010)).
- **Four × two demographics** — race (White / Black / Hispanic / Asian) × gender (Male / Female), signalled by validated names ([Elder & Hayes 2023](https://www.journals.uchicago.edu/doi/abs/10.1086/723820)).
- **Seven bundle types** — race, gender, and framing, each presented both transparently (a within-prompt minimal pair) and disguised (the focal axis co-varies with scenario), plus an intersectional race × gender bundle.

The accompanying paper gives the full benchmark design and the main results.

## Loading

Paths below are relative to the directory containing these files.

```r
library(readr)
library(dplyr)
outcomes <- read_csv("outcomes.csv")
stimuli  <- read_csv("stimuli.csv")
models   <- read_csv("models.csv")

# The reported analyses use the base wording variant:
base <- outcomes |> filter(wording_variant == "base", valid == TRUE)
```

```python
import pandas as pd
outcomes = pd.read_csv("outcomes.csv")
stimuli  = pd.read_csv("stimuli.csv")
models   = pd.read_csv("models.csv")
base     = outcomes.query("wording_variant == 'base' & valid")
```

The dataset also includes three additional wording-variant collections
(`need`, `merit`, `paraphrase`) alongside `base`. These are robustness
collections not analyzed in the accompanying paper; filter to `base` for the
reported analyses.

## Provenance

`outcomes.csv` is built directly from the raw model API responses, collected
at temperature 0 in April 2026. Parse coverage on the base wording variant is
high across all three tasks:

- Rate: 99.96% of rows parsed cleanly.
- Rank: 96.7%.
- Allocate: 99.1%.

Rows that could not be parsed are retained and flagged (`valid == FALSE`)
rather than dropped.

## References

- Elder, E. M., & Hayes, M. (2023). Signaling race, ethnicity, and gender with
  names: Challenges and recommendations. *The Journal of Politics*, 85(2),
  764–770. https://doi.org/10.1086/723820
- van Oorschot, W., & Roosma, F. (2017). The social legitimacy of targeted
  welfare and welfare deservingness. In W. van Oorschot, F. Roosma, B. Meuleman,
  & T. Reeskens (Eds.), *The Social Legitimacy of Targeted Welfare* (pp. 3–34).
  Edward Elgar. https://doi.org/10.4337/9781785367212.00010

## License

This dataset is released under CC-BY 4.0 (see the `LICENSE` file in this
directory). Names are drawn from Elder & Hayes (2023); please cite their work
if you build on the demographic-signalling component. Model outputs are
subject to each provider's terms of service. The repository's code is released
separately under the MIT License.
