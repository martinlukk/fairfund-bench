
## Program:   02-render_demographic_gap.R
## Task:      Render paper Figure 6: the mean absolute demographic
##            difference in Allocate dollars, transparent vs. disguised
##            bundles, averaged across the 14 models.
##
## Input:     data/outcomes.csv
## Output:    analysis/figures/fig06_demographic_gap.pdf
##
## Project:   fairfund-bench
## Author:    Martin Lukk / 2026-08-13 (created)

# 0. Program Setup --------------------------------------------------------
source(here::here("analysis/lib/common.R"))

# 1. Load Allocate Responses ----------------------------------------------
d <- load_outcomes() |>
  filter(task == "allocate", focal_axis %in% c("race", "gender"))

# 2. Per-Model Signed Gaps ------------------------------------------------
# Both axes are within-bundle contrasts, so each bundle supplies its own
# comparison and the per-model estimate is the mean of those differences.
race_gaps <- function(df) {
  df |>
    select(model, unit_id, race, dollars) |>
    pivot_wider(names_from = race, values_from = dollars) |>
    transmute(model, unit_id,
              `Black - White`    = Black    - White,
              `Hispanic - White` = Hispanic - White,
              `Asian - White`    = Asian    - White) |>
    pivot_longer(-c(model, unit_id),
                 names_to = "contrast", values_to = "delta") |>
    drop_na(delta) |>
    group_by(model, contrast) |>
    summarise(estimate = mean(delta), .groups = "drop")
}

gender_gaps <- function(df) {
  df |>
    select(model, unit_id, gender, dollars) |>
    pivot_wider(names_from = gender, values_from = dollars) |>
    transmute(model, unit_id, delta = Female - Male) |>
    drop_na(delta) |>
    group_by(model) |>
    summarise(estimate = mean(delta), .groups = "drop") |>
    mutate(contrast = "Female - Male")
}

gaps <- bind_rows(
  d |> filter(focal_axis == "race") |>
    group_by(mode) |> group_modify(~ race_gaps(.x)) |> ungroup() |>
    mutate(axis = "Race"),
  d |> filter(focal_axis == "gender") |>
    group_by(mode) |> group_modify(~ gender_gaps(.x)) |> ungroup() |>
    mutate(axis = "Gender")) |>
  mutate(abs_est = abs(estimate))

# 3. Across-Model Magnitudes ----------------------------------------------
class_means <- gaps |>
  group_by(axis, mode) |>
  summarise(mean_abs = mean(abs_est),
            lo = mean(abs_est) - 1.96 * sd(abs_est) / sqrt(n()),
            hi = mean(abs_est) + 1.96 * sd(abs_est) / sqrt(n()),
            .groups = "drop") |>
  mutate(axis = factor(axis, levels = c("Race", "Gender")),
         mode = factor(mode, levels = c("transparent", "disguised"),
                       labels = c("Transparent", "Disguised")))

print(class_means)

# 4. Figure ---------------------------------------------------------------
p <- ggplot(class_means, aes(x = axis, y = mean_abs, fill = mode)) +
  geom_col(position = position_dodge(0.7), width = 0.6) +
  geom_errorbar(aes(ymin = lo, ymax = hi),
                position = position_dodge(0.7),
                width = 0.18, color = "grey20") +
  scale_fill_manual(values = c(Transparent = "#a7c4e3",
                               Disguised   = "#3b6ea8"),
                    name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.08)),
                     labels = function(x) paste0("$", x)) +
  labs(x = NULL, y = NULL) +
  theme(legend.position = "bottom",
        legend.margin = margin(t = -4, b = 0),
        legend.key.size = unit(0.75, "lines"),
        axis.text.x = element_text(size = 9))

ggsave(file.path(FIG_DIR, "fig06_demographic_gap.pdf"), p,
       width = 3.3, height = 2.5, device = cairo_pdf)

cat("Wrote", file.path(FIG_DIR, "fig06_demographic_gap.pdf"), "\n")
