
## Program:   01-render_equal_split.R
## Task:      Render paper Figure 5: the share of Allocate bundles in which
##            every claimant receives the same dollar amount, by focal axis
##            and presentation mode, with one line per model.
##
## Input:     data/outcomes.csv
##            data/models.csv
## Output:    analysis/figures/fig05_equal_split.pdf
##            analysis/figures/fig05_equal_split_wide.png (README preview)
##
## Project:   fairfund-bench
## Author:    Martin Lukk / 2026-08-13 (created)

# 0. Program Setup --------------------------------------------------------
source(here::here("analysis/lib/common.R"))

# 1. Load Allocate Responses ----------------------------------------------
d <- load_outcomes() |>
  filter(task == "allocate", focal_axis %in% c("race", "gender", "framing"))
models <- load_models()

# 2. Per-Bundle Equal-Split Flag ------------------------------------------
# One row per (model, bundle). focal_axis and mode are constant within a
# bundle, so grouping on them carries them through to the summary.
bundles <- d |>
  group_by(model, focal_axis, mode, unit_id) |>
  summarise(equal_split = n_distinct(dollars) == 1L, .groups = "drop")

# 3. Per-Model Rates ------------------------------------------------------
rates <- bundles |>
  group_by(model, focal_axis, mode) |>
  summarise(equal_split_pct = 100 * mean(equal_split), .groups = "drop") |>
  left_join(select(models, model, display_name), by = "model") |>
  mutate(
    axis = factor(str_to_title(focal_axis),
                  levels = c("Race", "Gender", "Framing")),
    presentation = factor(mode,
                          levels = c("transparent", "disguised"),
                          labels = c("Transp.", "Disg.")),
    grp = factor(if_else(display_name == "Grok 4.20",
                         "Grok 4.20", "Other models"),
                 levels = c("Other models", "Grok 4.20")))

rates |>
  group_by(axis, presentation) |>
  summarise(median_pct = median(equal_split_pct),
            min_pct = min(equal_split_pct),
            max_pct = max(equal_split_pct), .groups = "drop") |>
  print(n = Inf)

# 4. Figure ---------------------------------------------------------------
# Grok 4.20 is highlighted because it is the one model that does not
# saturate on the transparent demographic axes.
p <- ggplot(rates, aes(x = presentation, y = equal_split_pct,
                       group = display_name)) +
  geom_line(aes(color = grp, linewidth = grp), alpha = 0.75) +
  geom_point(aes(color = grp, size = grp)) +
  scale_color_manual(values = c(`Other models` = "grey55",
                                `Grok 4.20`    = "#c1272d"),
                     name = NULL) +
  scale_linewidth_manual(values = c(`Other models` = 0.4,
                                    `Grok 4.20`    = 0.9),
                         guide = "none") +
  scale_size_manual(values = c(`Other models` = 1.2,
                               `Grok 4.20`    = 1.9),
                    guide = "none") +
  scale_y_continuous(limits = c(0, 100),
                     breaks = c(0, 25, 50, 75, 100),
                     labels = function(x) paste0(x, "%")) +
  facet_wrap(~ axis) +
  labs(x = NULL, y = NULL) +
  guides(color = guide_legend(
    override.aes = list(linewidth = c(0.4, 0.9), size = c(1.2, 1.9)))) +
  theme(axis.text.x      = element_text(size = 8),
        legend.position  = "bottom",
        legend.margin    = margin(t = -4),
        legend.text      = element_text(size = 8),
        legend.key.width = unit(0.6, "lines"))

ggsave(file.path(FIG_DIR, "fig05_equal_split.pdf"), p,
       width = 3.3, height = 2.9, device = cairo_pdf)

cat("Wrote", file.path(FIG_DIR, "fig05_equal_split.pdf"), "\n")

# 5. README Preview PNG ---------------------------------------------------
p_wide <- p +
  theme(text             = element_text(size = 11),
        axis.text.x      = element_text(size = 10),
        legend.text      = element_text(size = 10),
        legend.key.width = unit(0.9, "lines"))

ggsave(file.path(FIG_DIR, "fig05_equal_split_wide.png"), p_wide,
       width = 7, height = 4.2, dpi = 300, device = ragg::agg_png)

cat("Wrote", file.path(FIG_DIR, "fig05_equal_split_wide.png"), "\n")
