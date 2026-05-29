# FairFund-Bench

**Evaluating Deservingness Bias in LLM Allocation Decisions**

FairFund-Bench is a benchmark for evaluating demographic bias
in how large language models allocate scarce resources. It presents models
with realistic financial aid requests and systematically varies the audit
design — the elicitation task (rating, ranking, dollar allocation), the
comparison context (single vs. multi-stimulus), and whether demographic
differences are made transparent or disguised. Varying these characteristics shows how conclusions
about LLM bias depend on auditing choices.

The benchmark comprises 600 human-authored aid requests (calibrated against
1.3M real GoFundMe campaigns) spanning three need domains, four race and two
gender categories (signalled via validated names), and five causal framings
of need drawn from welfare deservingness theory. It scores models on four
pillars: demographic bias, normative alignment, cross-task consistency, and
cross-context consistency.

## Repository layout

```
data/        Model-response dataset and codebook (see data/README.md)
benchmark/   Benchmark instrument and collection code
analysis/    Reproduction scripts and figures for accompanying paper
```

## Data

The released dataset is in [`data/`](data/) and includes: the parsed responses from 14 LLMs
(`outcomes.csv`), the 3,000-stimulus universe with full appeal text
(`stimuli.csv`), and the model lineup (`models.csv`). See
[`data/codebook.md`](data/codebook.md) for column-by-column documentation and
[`data/README.md`](data/README.md) for loading instructions and provenance.

```r
library(readr)
library(dplyr)
outcomes <- read_csv("data/outcomes.csv")
# Analyses reported in the accompanying paper use the base wording variant:
base <- outcomes |> filter(wording_variant == "base", valid == TRUE)
```

```python
import pandas as pd
outcomes = pd.read_csv("data/outcomes.csv")
base = outcomes.query("wording_variant == 'base' & valid")
```

## License

- **Code** (this repository) is released under the MIT License (see `LICENSE`).
- **Data** (`data/`) is released under CC-BY 4.0 (see `data/LICENSE`).
- Names are drawn from [Elder & Hayes (2023)](https://www.journals.uchicago.edu/doi/abs/10.1086/723820); please cite their work if you
  build on the demographic-signalling component.
- Third-party model outputs are subject to each provider's terms of service;
  responses were collected at API temperature 0 in April 2026.
