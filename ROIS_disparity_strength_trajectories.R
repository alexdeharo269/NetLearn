# ================================================================
# NODAL DISPARITY RATIO ANALYSIS — GAM per ROI
# Column structure (from Python pd.concat of 4 dataframes):
#   R1..R90        = df_rois_disparities      (raw disparity Y)
#   R1.1..R90.1    = df_rois_strengths         (nodal strength)
#   R1.2..R90.2    = df_rois_disparities_ratio (Disparity_Ratio — USE THIS)
#   R1.3..R90.3    = df_rois_disparities_dev_std (deviation std)
#
# Model per node: Ratio_i ~ s(age) + sex + dataset + s(Strength_i)
# FDR (BH) across all 90 ROIs
# Output: procdata/Rcombat_harm/nodal/
# ================================================================
library(mgcv); library(ggplot2); library(dplyr); library(visreg)

out <- "C:/Users/Ale/Desktop/UB/NetLearn/procdata/Rcombat_harm/nodal/"
dir.create(out, showWarnings=FALSE, recursive=TRUE)

# ── Load & prepare ───────────────────────────────────────────────
d_full <- read.csv("C:/Users/Ale/Desktop/UB/NetLearn/procdata/demographics_with_rois_disparity_strength.csv")
d_full$sex <- as.factor(d_full$sex); d_full$dataset <- as.factor(d_full$dataset)

# Exclude datasets 1, 7, 8 (harmonization failures)
d <- droplevels(subset(d_full, !(dataset %in% c(1, 7, 8))))
cat(sprintf("N=%d | Datasets: %s | Age: %.1f–%.1f\n",
            nrow(d), paste(levels(d$dataset), collapse=","),
            min(d$age), max(d$age)))

# ── Identify column blocks ───────────────────────────────────────
all_cols <- colnames(d_full)

# Block 1: raw disparity — R1, R2, ..., R90 (no dot suffix)
disp_cols  <- grep("^R[0-9]+$",     all_cols, value=TRUE)

# Block 2: strength — R1.1, R2.1, ..., R90.1
str_cols   <- grep("^R[0-9]+\\.1$", all_cols, value=TRUE)

# Block 3: ratio — R1.2, R2.2, ..., R90.2  ← PRIMARY OUTCOME
ratio_cols <- grep("^R[0-9]+\\.2$", all_cols, value=TRUE)

# Block 4: deviation std — R1.3, R2.3, ..., R90.3
dev_cols   <- grep("^R[0-9]+\\.3$", all_cols, value=TRUE)

n_rois <- length(ratio_cols)
cat(sprintf("Disparity cols: %d | Strength cols: %d | Ratio cols: %d | DevStd cols: %d\n",
            length(disp_cols), length(str_cols), n_rois, length(dev_cols)))
stopifnot(n_rois > 0, length(str_cols) == n_rois)

# ROI index extractor (R1.2 → 1, R45.2 → 45)
roi_index <- function(col_name) as.integer(sub("^R([0-9]+).*", "\\1", col_name))

# Quick sanity: ratio values should mostly be > 1
ratio_sample <- as.numeric(as.matrix(d[1:min(200,nrow(d)), ratio_cols]))
ratio_sample <- ratio_sample[!is.na(ratio_sample)]
cat(sprintf("Ratio check (sample mean=%.3f, %% > 1 = %.1f%%)\n",
            mean(ratio_sample), 100*mean(ratio_sample > 1)))

# ── GAM per ROI ─────────────────────────────────────────────────
cat(sprintf("\nFitting %d GAMs (Ratio ~ s(age) + sex + dataset + s(Strength))...\n", n_rois))
results <- vector("list", n_rois)

for (i in seq_along(ratio_cols)) {
  rc  <- ratio_cols[i]   # e.g. "R5.2"
  sc  <- str_cols[i]     # e.g. "R5.1"
  idx <- roi_index(rc)
  
  y   <- d[[rc]]
  str <- d[[sc]]
  
  pct_valid <- mean(!is.na(y) & !is.na(str))
  if (pct_valid < 0.5) {
    results[[i]] <- data.frame(roi=rc, roi_idx=idx, n_valid=round(pct_valid*nrow(d)),
                               age_edf=NA, age_F=NA, age_p=NA, r2=NA, dev_expl=NA,
                               skipped=TRUE, reason="<50% valid"); next
  }
  
  d_roi <- data.frame(
    y_roi   = y,
    str_roi = str,
    age     = d$age,
    sex     = d$sex,
    dataset = d$dataset
  )
  d_roi <- droplevels(d_roi[!is.na(d_roi$y_roi) & !is.na(d_roi$str_roi), ])
  
  m <- tryCatch(
    gam(y_roi ~ s(age, bs="cr", k=8) + sex + dataset + s(str_roi, bs="cr", k=5),
        data=d_roi, method="REML"),
    error=function(e) NULL
  )
  
  if (is.null(m)) {
    results[[i]] <- data.frame(roi=rc, roi_idx=idx, n_valid=nrow(d_roi),
                               age_edf=NA, age_F=NA, age_p=NA, r2=NA, dev_expl=NA, skipped=TRUE, reason="gam_error")
  } else {
    sm      <- summary(m)
    age_row <- sm$s.table[grepl("s\\(age\\)", rownames(sm$s.table)),,drop=FALSE]
    results[[i]] <- data.frame(
      roi      = rc,
      roi_idx  = idx,
      n_valid  = nrow(d_roi),
      age_edf  = round(age_row[1,"edf"], 3),
      age_F    = round(age_row[1,"F"],   3),
      age_p    = age_row[1,"p-value"],
      r2       = round(sm$r.sq, 4),
      dev_expl = round(sm$dev.expl*100, 2),
      skipped  = FALSE,
      reason   = "ok"
    )
  }
  if (i %% 20 == 0) cat(sprintf("  %d/%d done\n", i, n_rois))
}

# ── FDR correction ───────────────────────────────────────────────
res_df <- do.call(rbind, results)
valid  <- res_df[!res_df$skipped, ]
valid$age_p_fdr <- p.adjust(valid$age_p, method="BH")
valid$sig_fdr   <- valid$age_p_fdr < 0.05
valid$sig_unc   <- valid$age_p     < 0.05
res_df <- merge(res_df, valid[, c("roi","age_p_fdr","sig_fdr","sig_unc")],
                by="roi", all.x=TRUE)
res_df <- res_df[order(res_df$roi_idx), ]

cat(sprintf("\nROIs fitted: %d | Skipped: %d\n",
            sum(!res_df$skipped), sum(res_df$skipped)))
cat(sprintf("Uncorrected p<0.05: %d (%.1f%%)\n",
            sum(res_df$sig_unc,na.rm=TRUE), 100*mean(res_df$sig_unc,na.rm=TRUE)))
cat(sprintf("FDR q<0.05:         %d (%.1f%%)\n",
            sum(res_df$sig_fdr,na.rm=TRUE), 100*mean(res_df$sig_fdr,na.rm=TRUE)))

# Append extra columns useful for brain map
res_df$neg_log10_p   <- -log10(res_df$age_p)
res_df$neg_log10_q   <- -log10(res_df$age_p_fdr)
res_df$sig_fdr_int   <- as.integer(res_df$sig_fdr)

write.csv(res_df, file.path(out,"nodal_gam_results.csv"), row.names=FALSE)

# Top 20
top20 <- head(res_df[order(-res_df$age_F, na.last=TRUE), ], 20)
cat("\nTop 20 nodes by F:\n")
print(top20[, c("roi","roi_idx","age_edf","age_F","age_p","age_p_fdr","sig_fdr","r2")])
write.csv(top20, file.path(out,"nodal_top20.csv"), row.names=FALSE)

# ================================================================
# PLOTS
# ================================================================

# A. Manhattan: F statistic per ROI
ggplot(res_df[!is.na(res_df$age_F),], aes(x=roi_idx, y=age_F, colour=sig_fdr)) +
  geom_point(size=1.8, alpha=0.85) +
  geom_hline(yintercept=quantile(res_df$age_F, 0.95, na.rm=TRUE),
             linetype="dashed", colour="grey40", linewidth=0.6) +
  scale_colour_manual(values=c("FALSE"="grey70","TRUE"="#E74C3C"), name="FDR q<0.05") +
  labs(title="Nodal Disparity_Ratio — age F statistic per ROI",
       subtitle="Red = FDR-significant | Dashed = 95th pctile",
       x="ROI index (1–90, AAL order)", y="F statistic [s(age)]") +
  theme_bw(base_size=12) + theme(plot.title=element_text(face="bold"))
ggsave(file.path(out,"01_manhattan_F.pdf"), width=14, height=5)

# B. Manhattan: -log10(FDR q)
fdr_thresh <- if (any(res_df$sig_fdr==TRUE, na.rm=TRUE))
  -log10(max(res_df$age_p[res_df$sig_fdr==TRUE], na.rm=TRUE)) else NA

ggplot(res_df[!is.na(res_df$age_p),], aes(x=roi_idx, y=neg_log10_p, colour=sig_fdr)) +
  geom_point(size=1.8, alpha=0.85) +
  geom_hline(yintercept=-log10(0.05), linetype="dotted", colour="orange", linewidth=0.7) +
  {if(!is.na(fdr_thresh))
    geom_hline(yintercept=fdr_thresh, linetype="dashed", colour="red", linewidth=0.7)} +
  scale_colour_manual(values=c("FALSE"="grey70","TRUE"="#E74C3C"), name="FDR q<0.05") +
  labs(title="Nodal p-value distribution across ROIs",
       subtitle="Orange = uncorrected p=0.05 | Red dashed = FDR threshold",
       x="ROI index", y="-log10(p)") +
  theme_bw(base_size=12) + theme(plot.title=element_text(face="bold"))
ggsave(file.path(out,"02_manhattan_pvalue.pdf"), width=14, height=5)

# C. EDF for significant nodes
sig_df <- res_df[!is.na(res_df$sig_fdr) & res_df$sig_fdr==TRUE, ]
if (nrow(sig_df) > 0) {
  ggplot(sig_df, aes(x=age_edf)) +
    geom_histogram(bins=20, fill="#2980B9", colour="white", alpha=0.85) +
    geom_vline(xintercept=1, linetype="dashed", colour="red") +
    labs(title="EDF distribution — FDR-significant nodes",
         subtitle="edf = 1: linear | edf > 1: non-linear age trajectory",
         x="edf [s(age)]", y="Count") +
    theme_bw(base_size=12) + theme(plot.title=element_text(face="bold"))
  ggsave(file.path(out,"03_edf_distribution.pdf"), width=8, height=5)
}

# D. R² distribution
ggplot(res_df[!is.na(res_df$r2), ], aes(x=r2, fill=sig_fdr)) +
  geom_histogram(bins=30, position="identity", alpha=0.7, colour="white") +
  scale_fill_manual(values=c("FALSE"="grey70","TRUE"="#E74C3C"), name="FDR q<0.05") +
  labs(title="Model R² across ROIs", x="R²", y="Count") +
  theme_bw(base_size=12) + theme(plot.title=element_text(face="bold"))
ggsave(file.path(out,"04_R2_distribution.pdf"), width=9, height=5)

# E. Top 9 node trajectories (partial effects)
top9_rois <- head(res_df[!is.na(res_df$sig_fdr) & res_df$sig_fdr==TRUE, ][
  order(-res_df$age_F[!is.na(res_df$sig_fdr) & res_df$sig_fdr==TRUE]), ], 9)$roi

if (length(top9_rois) > 0) {
  pdf(file.path(out,"05_top9_trajectories.pdf"), width=12, height=9)
  par(mfrow=c(3,3), mar=c(4,4,3.5,1), oma=c(0,0,3,0))
  for (rc in top9_rois) {
    sc    <- gsub("\\.2$", ".1", rc)   # R5.2 → R5.1
    d_roi <- data.frame(
      y_roi   = d[[rc]],
      str_roi = d[[sc]],
      age     = d$age,
      sex     = d$sex,
      dataset = d$dataset)
    d_roi <- droplevels(d_roi[!is.na(d_roi$y_roi) & !is.na(d_roi$str_roi), ])
    m <- tryCatch(
      gam(y_roi ~ s(age,bs="cr",k=8) + sex + dataset + s(str_roi,bs="cr",k=5),
          data=d_roi, method="REML"), error=function(e) NULL)
    if (!is.null(m)) {
      v <- visreg(m, "age", plot=FALSE)
      row <- res_df[res_df$roi==rc, ]
      plot(v$res$age, v$res$visregRes, pch=16, cex=0.35, col="grey75",
           xlab="Age (years)", ylab="Partial effect",
           main=sprintf("%s | edf=%.1f | q=%.3f",
                        sub("\\.2$","",rc), row$age_edf, row$age_p_fdr))
      lines(v$fit$age, v$fit$visregFit, col="#E74C3C", lwd=2)
      lines(v$fit$age, v$fit$visregLwr, col="#E74C3C", lwd=1, lty=2)
      lines(v$fit$age, v$fit$visregUpr, col="#E74C3C", lwd=1, lty=2)
      abline(h=0, col="grey40", lty=3)
    }
  }
  mtext("Top 9 FDR-significant nodes — Disparity_Ratio age trajectory",
        outer=TRUE, cex=1.1, font=2)
  dev.off()
}

cat(sprintf("\n=== DONE === FDR significant: %d / %d ROIs (%.1f%%)\n",
            sum(res_df$sig_fdr,na.rm=TRUE), n_rois,
            100*mean(res_df$sig_fdr,na.rm=TRUE)))
cat(sprintf("Brain map export: %s\n", file.path(out,"nodal_gam_results.csv")))

