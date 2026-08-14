
## Program:   03-render_framing.R
## Task:      Render paper Figure 7: mean Allocate dollars by causal
##            framing condition, pooled across the 14 models, on the
##            framing-transparent bundles.
##
## Input:     data/outcomes.csv
## Output:    analysis/figures/fig07_framing.pdf
##
## Project:   fairfund-bench
## Author:    Martin Lukk / 2026-08-13 (created)

# 0. Program Setup --------------------------------------------------------
source(here::here("analysis/lib/common.R"))
suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
})
emm_options(lmerTest.limit = 1e5, pbkrtest.limit = 1e5)

# 1. Load the Framing-Transparent Pool ------------------------------------
# These bundles hold scenario, race, gender, and category fixed and vary
# only the framing condition, so framing is the sole within-bundle contrast.
d <- load_outcomes() |>
  filter(task == "allocate", bundle_type == "framing_transparent")

# 2. Pooled Means by Framing ----------------------------------------------
fit <- suppressMessages(lmer(dollars ~ framing + (1 | model), data = d))

means <- emmeans(fit, ~ framing) |>
  as_tibble() |>
  transmute(framing = factor(framing,
                             levels = levels(d$framing),
                             labels = c("No cause", "Structural",
                                        "Self-cause", "Stigma",
                                        "Redemption")),
            mean = emmean, lo = lower.CL, hi = upper.CL)

print(means)

# 3. Figure ---------------------------------------------------------------
p <- ggplot(means, aes(x = framing, y = mean)) +
  geom_col(fill = "#3b6ea8", width = 0.7) +
  geom_errorbar(aes(ymin = lo, ymax = hi), width = 0.22, color = "grey20") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08)),
                     breaks = seq(0, 3000, by = 500),
                     labels = function(x) paste0("$", format(x, big.mark = ","))) +
  labs(x = NULL, y = NULL) +
  theme(axis.text.x = element_text(size = 8))

ggsave(file.path(FIG_DIR, "fig07_framing.pdf"), p,
       width = 3.3, height = 2.3, device = cairo_pdf)

cat("Wrote", file.path(FIG_DIR, "fig07_framing.pdf"), "\n")
