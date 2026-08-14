
## Program:   common.R
## Task:      Shared loader and plot theme for the figure scripts. Applies
##            the paper's reference factor levels (White, Male, Rate,
##            no_cause) so contrasts come out oriented as reported, and by
##            default keeps only base-wording, validly parsed rows.
##
## Input:     data/outcomes.csv
##            data/models.csv
## Output:    (none — sourced by the numbered scripts)
##
## Project:   fairfund-bench
## Author:    Martin Lukk / 2026-08-13 (created)

suppressPackageStartupMessages({
  library(tidyverse)
  library(here)
})

FIG_DIR <- here("analysis/figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

theme_set(theme_minimal(base_size = 9) +
            theme(panel.grid.minor = element_blank(),
                  plot.title.position = "plot",
                  strip.text = element_text(face = "bold")))

load_outcomes <- function(filter_base_valid = TRUE) {
  d <- read_csv(here("data/outcomes.csv"), show_col_types = FALSE,
                progress = FALSE)
  if (filter_base_valid) {
    d <- filter(d, wording_variant == "base", valid)
  }
  d |>
    mutate(race    = factor(race,
                            levels = c("White", "Black", "Hispanic", "Asian")),
           gender  = factor(gender, levels = c("Male", "Female")),
           framing = factor(framing,
                            levels = c("no_cause", "structural", "self_cause",
                                       "stigma_no_redemption",
                                       "stigma_redemption")),
           task    = factor(task, levels = c("rate", "rank", "allocate")),
           mode    = factor(mode, levels = c("transparent", "disguised")))
}

load_models <- function() {
  read_csv(here("data/models.csv"), show_col_types = FALSE, progress = FALSE)
}
