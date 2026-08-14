# FairFund-Bench — Paper figures

R scripts that reproduce the three data figures in the accompanying paper
from the released responses in `../data/`. The leaderboard in the paper comes from
`../benchmark/score.py`.

| Script | Paper figure | Output |
|---|---|---|
| `01-render_equal_split.R`     | Figure 5 | `figures/fig05_equal_split.pdf` |
| `02-render_demographic_gap.R` | Figure 6 | `figures/fig06_demographic_gap.pdf` |
| `03-render_framing.R`         | Figure 7 | `figures/fig07_framing.pdf` |

`lib/common.R` is sourced by all three analysis scripts. It reads `../data/outcomes.csv`,
filters to `wording_variant == "base" & valid`, and sets the appropriate factor
reference levels (White, Male, Rate, `no_cause`).

## Run

From the repository root:

```bash
Rscript analysis/01-render_equal_split.R
Rscript analysis/02-render_demographic_gap.R
Rscript analysis/03-render_framing.R
```

Each script prints the quantities behind its figure and writes a PDF to
`analysis/figures/`. Requires `tidyverse` and `here`, plus a cairo-capable R
build for `cairo_pdf`; script 03 additionally needs `lme4`, `lmerTest`, and
`emmeans`.
