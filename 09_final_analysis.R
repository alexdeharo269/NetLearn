# ================================================================
# FINAL R ANALYSIS — Disparity Ratio, Option B, and Modularity
# Input: demographics_with_disparity_dev_unthresholded.csv
#        (produced by notebook_disparity_modularity.py)
#        Contains: Disparity_Y, Y_null, Disparity_Ratio,
#                  modularity_Q, n_communities, Density, age, sex, dataset
# Excludes: datasets 1, 2, 7, 8 (scanner artefacts + harmonization failures)
# Output: procdata/Rcombat_harm/final_analysis/
# ================================================================
library(mgcv); library(visreg); library(ggplot2); library(dplyr); library(tidyr)

out <- "C:/Users/Ale/Desktop/UB/NetLearn/procdata/Rcombat_harm/final_analysis/"
dir.create(out, showWarnings=FALSE, recursive=TRUE)
gg <- function(f, p, w=11, h=5) ggsave(file.path(out, f), p, width=w, height=h)

# ── Load & prepare ────────────────────────────────────────────────
d_all <- read.csv("C:/Users/Ale/Desktop/UB/NetLearn/procdata/demographics_with_disparity_dev_unthresholded.csv")
d_all$sex <- as.factor(d_all$sex); d_all$dataset <- as.factor(d_all$dataset)

EXCL_DS <- c(1,2,6)
d <- droplevels(subset(d_all, !(dataset %in% EXCL_DS)))
cat(sprintf("N=%d | Datasets: %s | Age: %.1f–%.1f\n",
            nrow(d), paste(levels(d$dataset), collapse=","),
            min(d$age), max(d$age)))

# Check required columns
required <- c("Disparity_Y","Y_null","Disparity_Ratio","Density")
have_mod <- "modularity_Q" %in% colnames(d_all)
missing  <- setdiff(required, colnames(d))
if (length(missing) > 0) stop(sprintf("Missing columns: %s", paste(missing, collapse=", ")))

cat(sprintf("\nDisparity_Y:     mean=%.4f  SD=%.4f\n",
            mean(d$Disparity_Y, na.rm=TRUE), sd(d$Disparity_Y, na.rm=TRUE)))
cat(sprintf("Y_null:          mean=%.4f  SD=%.4f\n",
            mean(d$Y_null, na.rm=TRUE), sd(d$Y_null, na.rm=TRUE)))
cat(sprintf("Disparity_Ratio: mean=%.3f  SD=%.3f  %%>1=%.0f%%\n",
            mean(d$Disparity_Ratio, na.rm=TRUE),
            sd(d$Disparity_Ratio, na.rm=TRUE),
            100*mean(d$Disparity_Ratio > 1, na.rm=TRUE)))
if (have_mod)
  cat(sprintf("Modularity_Q:    mean=%.3f  SD=%.3f\n",
              mean(d$modularity_Q[!d$dataset %in% EXCL_DS], na.rm=TRUE),
              sd(d$modularity_Q[!d$dataset %in% EXCL_DS], na.rm=TRUE)))

# ================================================================
# SECTION A — DISPARITY RATIO: main model
# ================================================================
cat("\n\n╔══════════════════════════════════════╗\n")
cat("║  A. DISPARITY RATIO MAIN MODEL       ║\n")
cat("╚══════════════════════════════════════╝\n")

# A1. Harmonization check
m_harm <- gam(Disparity_Ratio ~ s(age, by=dataset, bs="cr", k=4) +
                sex + dataset, data=d, method="REML")
age_harm <- summary(m_harm)$s.table[grepl("s\\(age\\)", rownames(summary(m_harm)$s.table)),]
n_sig_harm <- sum(age_harm[,"p-value"] < 0.05)
cat(sprintf("Harmonization check: %d/%d significant\n", n_sig_harm, nrow(age_harm)))
print(round(age_harm[, c("edf","F","p-value")], 4))

# A2. AIC: with vs without density (ratio already absorbs degree)
m_ratio_dens  <- gam(Disparity_Ratio ~ s(age,bs="cr",k=8) + sex + dataset +
                       s(Density,bs="cr",k=5), data=d, method="REML")
m_ratio_nodns <- gam(Disparity_Ratio ~ s(age,bs="cr",k=8) + sex + dataset,
                     data=d, method="REML")
delta_dens <- AIC(m_ratio_nodns) - AIC(m_ratio_dens)
use_density <- delta_dens < -2
cat(sprintf("\nAIC with Density=%.1f | without=%.1f | ΔAIC=%.1f → Density %s\n",
            AIC(m_ratio_dens), AIC(m_ratio_nodns), delta_dens,
            ifelse(use_density,"KEPT","DROPPED")))

m_ratio <- if (use_density) m_ratio_dens else m_ratio_nodns
sm_r    <- summary(m_ratio)
cat(sprintf("Final: R²=%.4f  Dev=%.2f%%  AIC=%.1f\n",
            sm_r$r.sq, sm_r$dev.expl*100, AIC(m_ratio)))
print(sm_r$s.table); print(sm_r$p.table)

m_null_r <- gam(Disparity_Ratio ~ sex + dataset, data=d, method="REML")
r_raw_r  <- cor.test(d$age, d$Disparity_Ratio)
r_part_r <- cor.test(d$age, residuals(m_null_r))
cat(sprintf("Ratio: raw r=%.4f | partial r=%.4f (p=%.3e)\n",
            r_raw_r$estimate, r_part_r$estimate, r_part_r$p.value))

# ================================================================
# SECTION B — OPTION B: Y_obs vs Y_null across age
# ================================================================
cat("\n\n╔══════════════════════════════════════╗\n")
cat("║  B. Y_obs vs Y_null RELATIONSHIP     ║\n")
cat("╚══════════════════════════════════════╝\n")

# B1. How strongly does Y_obs correlate with Y_null?
r_obs_null <- cor.test(d$Disparity_Y, d$Y_null)
lm_track   <- lm(Disparity_Y ~ Y_null, data=d)
cat(sprintf("cor(Y_obs, Y_null) = %.4f  p=%.3e\n",
            r_obs_null$estimate, r_obs_null$p.value))
cat(sprintf("lm(Y_obs~Y_null):  intercept=%.5f  slope=%.4f  R²=%.4f\n",
            coef(lm_track)[1], coef(lm_track)[2], summary(lm_track)$r.squared))

# B2. Does Y_obs show an age effect independent of Y_null?
m_yobs        <- gam(Disparity_Y ~ s(age,bs="cr",k=8) + sex + dataset,
                      data=d, method="REML")
m_yobs_ynull  <- gam(Disparity_Y ~ s(age,bs="cr",k=8) + sex + dataset + Y_null,
                      data=d, method="REML")
cat(sprintf("\nY_obs ~ s(age):           R²=%.4f  AIC=%.1f\n",
            summary(m_yobs)$r.sq, AIC(m_yobs)))
cat(sprintf("Y_obs ~ s(age) + Y_null:  R²=%.4f  AIC=%.1f  ΔAIC=%.1f → Y_null %s additional variance\n",
            summary(m_yobs_ynull)$r.sq, AIC(m_yobs_ynull),
            AIC(m_yobs_ynull) - AIC(m_yobs),
            ifelse(AIC(m_yobs_ynull) - AIC(m_yobs) < -2, "EXPLAINS","does NOT explain")))

# B3. Mean Ratio per age group
age_bins <- c(0,10,20,30,40,50,60,70,100)
age_lbls <- c("0-10","10-20","20-30","30-40","40-50","50-60","60-70","70+")
d$age_group <- cut(d$age, breaks=age_bins, labels=age_lbls, right=FALSE)
age_summary <- d %>%
  group_by(age_group) %>%
  summarise(n=n(), mean_Y_obs=round(mean(Disparity_Y,na.rm=TRUE),5),
            mean_Y_null=round(mean(Y_null,na.rm=TRUE),5),
            mean_Ratio=round(mean(Disparity_Ratio,na.rm=TRUE),4),
            sd_Ratio=round(sd(Disparity_Ratio,na.rm=TRUE),4),
            cor_obs_null=round(cor(Disparity_Y,Y_null,use="complete.obs"),4),
            .groups="drop")
cat("\nY_obs, Y_null, Ratio and correlation by age group:\n")
print(as.data.frame(age_summary))
write.csv(age_summary, file.path(out,"option_b_age_summary.csv"), row.names=FALSE)

# B4. GAM on Y_obs and Y_null trajectories for overlay plot
age_seq <- seq(1, 90, by=0.5)
# Modal dataset for prediction
modal_ds <- names(which.max(table(d$dataset)))
new_base <- data.frame(age=age_seq,
                       sex=factor(levels(d$sex)[1], levels=levels(d$sex)),
                       dataset=factor(modal_ds, levels=levels(d$dataset)),
                       Density=mean(d$Density), Y_null=mean(d$Y_null))

pm_yobs  <- predict(m_yobs,  newdata=new_base, se.fit=TRUE)
# Use a simpler model for Y_null trajectory
m_ynull  <- gam(Y_null ~ s(age,bs="cr",k=8) + sex + dataset, data=d, method="REML")
pm_ynull <- predict(m_ynull, newdata=new_base, se.fit=TRUE)

traj_both <- rbind(
  data.frame(age=age_seq, fit=pm_yobs$fit,  lo=pm_yobs$fit-2*pm_yobs$se.fit,
             hi=pm_yobs$fit+2*pm_yobs$se.fit,  metric="Y_obs"),
  data.frame(age=age_seq, fit=pm_ynull$fit, lo=pm_ynull$fit-2*pm_ynull$se.fit,
             hi=pm_ynull$fit+2*pm_ynull$se.fit, metric="Y_null = 2/(k+1)"))

write.csv(traj_both, file.path(out,"yobs_ynull_trajectories.csv"), row.names=FALSE)

# ================================================================
# SECTION C — MODULARITY
# ================================================================
if (have_mod) {
  cat("\n\n╔══════════════════════════════════════╗\n")
  cat("║  C. MODULARITY Q LIFESPAN GAM        ║\n")
  cat("╚══════════════════════════════════════╝\n")

  dm <- droplevels(d[!is.na(d$modularity_Q), ])
  cat(sprintf("N=%d | Q range: %.3f–%.3f\n",
              nrow(dm), min(dm$modularity_Q), max(dm$modularity_Q)))

  m_mod  <- gam(modularity_Q ~ s(age,bs="cr",k=8) + sex + dataset, data=dm, method="REML")
  sm_mod <- summary(m_mod)
  cat(sprintf("Modularity: R²=%.4f  Dev=%.2f%%  AIC=%.1f\n",
              sm_mod$r.sq, sm_mod$dev.expl*100, AIC(m_mod)))
  print(sm_mod$s.table); print(sm_mod$p.table)

  m_null_mod <- gam(modularity_Q ~ sex + dataset, data=dm, method="REML")
  r_mod  <- cor.test(dm$age, residuals(m_null_mod))
  cat(sprintf("Modularity partial r=%.4f  p=%.3e\n", r_mod$estimate, r_mod$p.value))

  # Sex-specific
  m_mod_F <- gam(modularity_Q ~ s(age,bs="cr",k=8) + dataset,
                 data=droplevels(subset(dm,sex==1)), method="REML")
  m_mod_M <- gam(modularity_Q ~ s(age,bs="cr",k=8) + dataset,
                 data=droplevels(subset(dm,sex==0)), method="REML")
  cat("Female modularity:\n"); print(summary(m_mod_F)$s.table)
  cat("Male modularity:\n");   print(summary(m_mod_M)$s.table)

  # Modularity predictions
  pm_mod <- predict(m_mod, newdata=new_base[new_base$age %in% seq(min(dm$age),max(dm$age),0.5),],
                    se.fit=TRUE)
  age_mod_seq <- seq(min(dm$age), max(dm$age), by=0.5)
  new_mod <- data.frame(age=age_mod_seq,
    sex=factor(levels(dm$sex)[1], levels=levels(dm$sex)),
    dataset=factor(modal_ds, levels=levels(dm$dataset)))
  pm_mod <- predict(m_mod, newdata=new_mod, se.fit=TRUE)
  pred_mod <- data.frame(age=age_mod_seq, fit=pm_mod$fit,
    ci_lo=pm_mod$fit-2*pm_mod$se.fit, ci_hi=pm_mod$fit+2*pm_mod$se.fit)
  write.csv(pred_mod, file.path(out,"modularity_predictions.csv"), row.names=FALSE)

  # Harmonization check for modularity
  m_mod_harm <- gam(modularity_Q ~ s(age,by=dataset,bs="cr",k=4) + sex + dataset,
                    data=dm, method="REML")
  age_mod_h  <- summary(m_mod_harm)$s.table[
    grepl("s\\(age\\)", rownames(summary(m_mod_harm)$s.table)),]
  n_sig_mod  <- sum(age_mod_h[,"p-value"] < 0.05)
  cat(sprintf("Modularity harmonization: %d/%d sig\n", n_sig_mod, nrow(age_mod_h)))

  # Visreg for modularity
  v_mod <- visreg(m_mod, "age", plot=FALSE)
  write.csv(data.frame(age=v_mod$res$age, Q_res=v_mod$res$visregRes,
                       dataset=dm$dataset, sex=dm$sex),
            file.path(out,"modularity_residuals.csv"), row.names=FALSE)
}

# ================================================================
# PLOTS
# ================================================================

# Plot 1: Y_obs and Y_null overlaid vs age (the key Option B plot)
gg("01_yobs_ynull_overlay.pdf",
   ggplot(traj_both, aes(x=age, y=fit, colour=metric, fill=metric)) +
     geom_ribbon(aes(ymin=lo, ymax=hi), alpha=0.15, colour=NA) +
     geom_line(linewidth=1.3) +
     scale_colour_manual(values=c("Y_obs"="#E74C3C", "Y_null = 2/(k+1)"="#2980B9")) +
     scale_fill_manual(  values=c("Y_obs"="#E74C3C", "Y_null = 2/(k+1)"="#2980B9")) +
     labs(title="Y_obs and Y_null as functions of age",
          subtitle="Parallel curves → Ratio is flat by construction | Diverging → real age effect on Ratio",
          x="Age (years)", y="Value", colour="", fill="") +
     theme_bw(base_size=12) + theme(legend.position="bottom",
                                     plot.title=element_text(face="bold")), w=11, h=6)

# Plot 2: Ratio flat across age groups
gg("02_ratio_by_age_group.pdf",
   ggplot(age_summary, aes(x=age_group, y=mean_Ratio)) +
     geom_col(fill="#27AE60", alpha=0.85, width=0.65) +
     geom_errorbar(aes(ymin=mean_Ratio-sd_Ratio, ymax=mean_Ratio+sd_Ratio),
                   width=0.25, colour="grey30") +
     geom_hline(yintercept=1, linetype="dashed", colour="red", linewidth=0.8) +
     labs(title="Mean Disparity_Ratio by age group",
          subtitle="Flat line = ratio is age-invariant | Red dashed = null expectation (ratio=1)",
          x="Age group", y="Mean Disparity_Ratio (±SD)") +
     theme_bw(base_size=12) + theme(plot.title=element_text(face="bold")))

# Plot 3: Y_obs vs Y_null scatter coloured by age
gg("03_yobs_vs_ynull_scatter.pdf",
   ggplot(d, aes(x=Y_null, y=Disparity_Y, colour=age)) +
     geom_point(alpha=0.2, size=0.8) +
     geom_smooth(method="lm", se=FALSE, colour="red", linewidth=1.2) +
     geom_abline(slope=1, intercept=0, linetype="dashed", colour="grey40",
                 linewidth=0.8) +
     scale_colour_viridis_c(option="plasma", name="Age") +
     labs(title=sprintf("Y_obs vs Y_null  (r=%.3f, slope=%.3f, R²=%.3f)",
                        r_obs_null$estimate, coef(lm_track)[2],
                        summary(lm_track)$r.squared),
          subtitle="Red = linear fit | Dashed = 1:1 line (Y_obs = Y_null)",
          x="Y_null = 2/(k+1)", y="Y_obs (raw disparity)") +
     theme_bw(base_size=12) + theme(plot.title=element_text(face="bold")), w=9, h=6)

# Plot 4: Ratio scatter + GAM fit
pred_ratio <- data.frame(age=age_seq,
  sex=factor(levels(d$sex)[1], levels=levels(d$sex)),
  dataset=factor(modal_ds, levels=levels(d$dataset)),
  Density=mean(d$Density))
pr <- predict(m_ratio, newdata=pred_ratio, se.fit=TRUE)
pred_ratio$fit <- pr$fit; pred_ratio$ci_lo <- pr$fit-2*pr$se.fit; pred_ratio$ci_hi <- pr$fit+2*pr$se.fit

gg("04_ratio_vs_age.pdf",
   ggplot(d, aes(x=age, y=Disparity_Ratio)) +
     geom_point(aes(colour=dataset), alpha=0.15, size=0.7) +
     geom_hline(yintercept=1, linetype="dashed", colour="red", linewidth=0.7) +
     geom_ribbon(data=pred_ratio, aes(y=fit,ymin=ci_lo,ymax=ci_hi),
                 fill="black", alpha=0.12) +
     geom_line(data=pred_ratio, aes(y=fit), colour="black", linewidth=1.3) +
     labs(title=sprintf("Disparity_Ratio vs Age  (partial r=%.3f, p=%.3f)",
                        r_part_r$estimate, r_part_r$p.value),
          subtitle="Red dashed = null expectation (Ratio = 1)",
          x="Age (years)", y="Disparity_Ratio", colour="Dataset") +
     theme_bw(base_size=12) + theme(plot.title=element_text(face="bold")), w=11, h=6)

# Plot 5: Harmonization check (base R)
suppressWarnings({
  pdf(file.path(out,"05_harmonization_check.pdf"), width=12, height=8)
  par(mfrow=c(ceiling(nlevels(d$dataset)/3),3), mar=c(4,4,3,1), oma=c(0,0,3.5,0))
  plot(m_harm, shade=TRUE, shade.col="lightyellow", seWithMean=TRUE, pages=0,
       xlab="Age", ylab="Partial effect on Ratio", cex.lab=1.1)
  mtext(sprintf("Harmonization: %d/%d sig", n_sig_harm, nrow(age_harm)),
        outer=TRUE, cex=1.1, font=2)
  dev.off()

  # Plot 6: Ratio age + density effects
  pdf(file.path(out,"06_ratio_model_effects.pdf"), width=11, height=5)
  par(mfrow=c(1, if(use_density) 2 else 1), mar=c(4.5,4.5,3,1), oma=c(0,0,3,0))
  plot(m_ratio, shade=TRUE, shade.col="lightblue", seWithMean=TRUE, pages=0,
       xlab=if(use_density) c("Age (years)","Density") else "Age (years)",
       ylab="Partial effect on Disparity_Ratio", cex.lab=1.2)
  mtext("Disparity_Ratio: marginal effects", outer=TRUE, cex=1.1, font=2)
  dev.off()
})

# Plot 7: Modularity
if (have_mod) {
  gg("07_modularity_vs_age.pdf",
     ggplot(dm, aes(x=age, y=modularity_Q)) +
       geom_point(aes(colour=dataset), alpha=0.18, size=0.7) +
       geom_ribbon(data=pred_mod, aes(y=fit,ymin=ci_lo,ymax=ci_hi),
                   fill="black", alpha=0.12) +
       geom_line(data=pred_mod, aes(y=fit), colour="black", linewidth=1.3) +
       labs(title=sprintf("Modularity Q vs Age  (partial r=%.3f, p=%.3e)",
                          r_mod$estimate, r_mod$p.value),
            subtitle=sprintf("R²=%.4f  Dev.expl=%.2f%%  edf=%.2f",
                             sm_mod$r.sq, sm_mod$dev.expl*100,
                             sm_mod$s.table[1,"edf"]),
            x="Age (years)", y="Modularity Q", colour="Dataset") +
       theme_bw(base_size=12) + theme(plot.title=element_text(face="bold")), w=11, h=6)

  suppressWarnings({
    pdf(file.path(out,"08_modularity_harmonization.pdf"), width=12, height=8)
    par(mfrow=c(ceiling(nlevels(dm$dataset)/3),3), mar=c(4,4,3,1), oma=c(0,0,3.5,0))
    plot(m_mod_harm, shade=TRUE, shade.col="lightyellow", seWithMean=TRUE, pages=0,
         xlab="Age", ylab="Partial effect on Q", cex.lab=1.1)
    mtext(sprintf("Modularity harmonization: %d/%d sig", n_sig_mod, nrow(age_mod_h)),
          outer=TRUE, cex=1.1, font=2)
    dev.off()
  })
}

# In your 09_final_analysis.R, after fitting m_ratio, add:
# Visualize the actual GAM smooth shape
pdf(file.path(out,"04b_ratio_gam_smooth.pdf"), width=9, height=5)
par(mar=c(4.5, 4.5, 3, 1))
plot(m_ratio, select=1, shade=TRUE, shade.col="lightgreen",
     seWithMean=TRUE,
     xlab="Age (years)", ylab="Partial effect on Disparity_Ratio",
     main="GAM smooth: age effect on Ratio (U-shape expected)",
     cex.lab=1.2)
abline(h=0, lty=2, col="grey50")
dev.off()

# Also: quadratic partial correlation to capture non-linearity
d$age2 <- d$age^2
m_quad <- lm(residuals(gam(Disparity_Ratio ~ sex + dataset,
                           data=d, method="REML")) ~ d$age + d$age2)
cat(sprintf("Quadratic age term: p=%.4f\n", summary(m_quad)$coef[3,4]))

# ================================================================
# SUMMARY
# ================================================================
cat("\n╔═══════════════════════════════════════════════╗\n")
cat("║  SUMMARY OF RESULTS                           ║\n")
cat("╚═══════════════════════════════════════════════╝\n")
cat(sprintf("Datasets excluded: %s\n", paste(EXCL_DS, collapse=",")))
cat(sprintf("N analysed: %d | Age: %.1f–%.1f\n", nrow(d), min(d$age), max(d$age)))
cat(sprintf("\nDisparity_Ratio: partial r=%.4f (p=%.3e) | R²=%.4f\n",
            r_part_r$estimate, r_part_r$p.value, sm_r$r.sq))
cat(sprintf("cor(Y_obs,Y_null)=%.4f  slope=%.4f  R²=%.4f\n",
            r_obs_null$estimate, coef(lm_track)[2], summary(lm_track)$r.squared))
if (have_mod)
  cat(sprintf("Modularity_Q:    partial r=%.4f (p=%.3e) | R²=%.4f\n",
              r_mod$estimate, r_mod$p.value, sm_mod$r.sq))
cat(sprintf("\nOutputs: %s\n", out))

