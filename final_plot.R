# ================================================================
# FINAL PLOT — Disparity_Ratio_B lifespan trajectory
# Result: NEGATIVE linear age effect after harmonization
# ================================================================
library(mgcv); library(ggplot2); library(dplyr); library(visreg)
if (!requireNamespace("patchwork",    quietly=TRUE)) install.packages("patchwork")
if (!requireNamespace("RColorBrewer", quietly=TRUE)) install.packages("RColorBrewer")
library(patchwork); library(RColorBrewer)

out <- "C:/Users/Ale/Desktop/UB/NetLearn/procdata/Rcombat_harm/age_stratified/"

d <- read.csv(file.path(out,"data_harmonized.csv"))
d$sex <- as.factor(d$sex); d$dataset <- as.factor(d$dataset)
METRIC   <- "Disparity_Ratio_B"
modal_ds <- names(which.max(table(d$dataset)))

# ── Fit models ────────────────────────────────────────────────────
m_final <- gam(as.formula(sprintf("%s ~ s(age,bs='cr',k=8) + sex + dataset", METRIC)),
               data=d, method="REML")
sf <- summary(m_final)

# THIS IS THE MISSING LINE — must come right after fitting m_final
v_main <- visreg(m_final, "age", plot=FALSE)

# Confirm direction from visreg (not from predict with modal_ds)
cat(sprintf("visreg age range: %.1f – %.1f\n",
            min(v_main$fit$age), max(v_main$fit$age)))
cat(sprintf("visreg fit at min age: %.4f | at max age: %.4f | Direction: %s\n",
            v_main$fit$visregFit[1],
            v_main$fit$visregFit[nrow(v_main$fit)],
            ifelse(v_main$fit$visregFit[nrow(v_main$fit)] >
                     v_main$fit$visregFit[1], "POSITIVE", "NEGATIVE")))

# Sex-specific models (named data objects — required by visreg)
d_F <- droplevels(d[d$sex == 1, ])
d_M <- droplevels(d[d$sex == 0, ])
m_F <- gam(as.formula(sprintf("%s ~ s(age,bs='cr',k=8) + dataset", METRIC)),
           data=d_F, method="REML")
m_M <- gam(as.formula(sprintf("%s ~ s(age,bs='cr',k=8) + dataset", METRIC)),
           data=d_M, method="REML")
v_F <- visreg(m_F, "age", plot=FALSE)
v_M <- visreg(m_M, "age", plot=FALSE)

# ── Stats strings ─────────────────────────────────────────────────
stats_str <- sprintf(
  "GAM: edf = %.2f,  F = %.2f,  p = %.3f\nR\u00b2 = %.3f,  N = %d",
  sf$s.table[1,"edf"], sf$s.table[1,"F"],
  sf$s.table[1,"p-value"], sf$r.sq, nrow(d))

sex_stats_F <- sprintf("F=%.2f, p=%.3f",
                       summary(m_F)$s.table[1,"F"],
                       summary(m_F)$s.table[1,"p-value"])
sex_stats_M <- sprintf("F=%.2f, p=%.3f",
                       summary(m_M)$s.table[1,"F"],
                       summary(m_M)$s.table[1,"p-value"])

ds_cols <- setNames(
  RColorBrewer::brewer.pal(max(3, nlevels(d$dataset)), "Set2")[seq_len(nlevels(d$dataset))],
  levels(d$dataset))

# ── PLOT 1: Main scatter + visreg GAM fit ─────────────────────────
p1 <- ggplot(d, aes(x=age, y=.data[[METRIC]])) +
  geom_point(aes(colour=dataset), alpha=0.18, size=0.65) +
  geom_ribbon(data=v_main$fit,
              aes(x=age, ymin=visregLwr, ymax=visregUpr),
              fill="#2C3E50", alpha=0.25, inherit.aes=FALSE) +
  geom_line(data=v_main$fit,
            aes(x=age, y=visregFit),
            colour="#2C3E50", linewidth=1.8, inherit.aes=FALSE) +
  annotate("text", x=12, y=1.625,
           label=stats_str, hjust=0, vjust=1,
           size=3.4, family="mono", colour="grey25") +
  scale_colour_manual(values=ds_cols, name="Dataset") +
  scale_x_continuous(breaks=seq(0, 90, 10)) +
  coord_cartesian(ylim=c(1.35, 1.65)) +
  labs(
    title="Network Disparity Ratio decreases across the adult lifespan",
    subtitle=sprintf(
      "Ratio = Y_obs / Y_null | Age-stratified harmonization (0/6 datasets sig) | edf=%.2f  F=%.2f  p=%.3f  N=%d",
      sf$s.table[1,"edf"], sf$s.table[1,"F"],
      sf$s.table[1,"p-value"], nrow(d)),
    x="Age (years)",
    y="Disparity Ratio  (Y_obs / Y_null)"
  ) +
  guides(colour=guide_legend(override.aes=list(alpha=0.8, size=2))) +
  theme_bw(base_size=13) +
  theme(plot.title    = element_text(face="bold", size=14),
        plot.subtitle = element_text(size=9, colour="grey40"),
        panel.grid.minor = element_blank())

# ── PLOT 2: Sex-specific partial effects ──────────────────────────
sex_df <- rbind(
  data.frame(age=v_F$fit$age, fit=v_F$fit$visregFit,
             lo=v_F$fit$visregLwr, hi=v_F$fit$visregUpr, Sex="Female"),
  data.frame(age=v_M$fit$age, fit=v_M$fit$visregFit,
             lo=v_M$fit$visregLwr, hi=v_M$fit$visregUpr, Sex="Male"))

p2 <- ggplot(sex_df, aes(x=age, y=fit, colour=Sex, fill=Sex)) +
  geom_ribbon(aes(ymin=lo, ymax=hi), alpha=0.18, colour=NA) +
  geom_line(linewidth=1.4) +
  annotate("text", x=72, y=max(sex_df$hi, na.rm=TRUE)*0.99,
           label=paste("Female:", sex_stats_F),
           colour="#C0392B", size=3.5, hjust=0) +
  annotate("text", x=72, y=max(sex_df$hi, na.rm=TRUE)*0.90,
           label=paste("Male:", sex_stats_M),
           colour="#2980B9", size=3.5, hjust=0) +
  scale_colour_manual(values=c(Female="#C0392B", Male="#2980B9")) +
  scale_fill_manual(  values=c(Female="#C0392B", Male="#2980B9")) +
  scale_x_continuous(breaks=seq(0, 90, 10)) +
  labs(
    title="Partial age effect by sex (controlling for dataset)",
    subtitle="Partial effect after dataset correction | 95% CI (2\u03c3)",
    x="Age (years)", y="Partial effect on Disparity Ratio"
  ) +
  theme_bw(base_size=13) +
  theme(plot.title    = element_text(face="bold", size=13),
        plot.subtitle = element_text(size=10, colour="grey40"),
        legend.position = "top",
        panel.grid.minor = element_blank())

# ── Save ──────────────────────────────────────────────────────────
ggsave(file.path(out,"FINAL_01_main_scatter.pdf"), p1, width=12, height=6)
ggsave(file.path(out,"FINAL_01_main_scatter.png"), p1, width=12, height=6, dpi=200)
ggsave(file.path(out,"FINAL_02_sex_trajectories.pdf"), p2, width=10, height=5)
ggsave(file.path(out,"FINAL_02_sex_trajectories.png"), p2, width=10, height=5, dpi=200)

p_combined <- (p1 / p2) +
  plot_annotation(
    caption=sprintf(
      "Datasets 3,4,5,7,8,9 | N=%d | Factor-smooth age-stratified harmonization | 0/6 datasets sig | %s",
      nrow(d), format(Sys.Date(),"%B %Y")),
    theme=theme(plot.caption=element_text(size=9, colour="grey50"))) +
  plot_layout(heights=c(1.4, 1))

ggsave(file.path(out,"FINAL_combined_panel.pdf"), p_combined, width=12, height=11)
ggsave(file.path(out,"FINAL_combined_panel.png"), p_combined, width=12, height=11, dpi=200)
cat("Done. Check console for direction confirmation.\n")