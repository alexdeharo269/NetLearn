# ================================================================
# NODAL LONG-FORMAT GAM
# Each observation = one node from one subject (N × 90 rows)
# s(subject_id, bs="re") = random intercept per subject
#   → accounts for within-subject node correlation
# Output: procdata/Rcombat_harm/nodal_long/
# ================================================================
library(mgcv); library(ggplot2); library(dplyr)

out <- "C:/Users/Ale/Desktop/UB/NetLearn/procdata/Rcombat_harm/nodal_long/"
dir.create(out, showWarnings=FALSE, recursive=TRUE)

# ── Load ──────────────────────────────────────────────────────────
cat("Loading nodal long-format data...\n")
d <- read.csv("C:/Users/Ale/Desktop/UB/NetLearn/procdata/nodal_long_format.csv")
d$sex        <- as.factor(d$sex)
d$dataset    <- as.factor(d$dataset)
d$subject_id <- as.factor(d$subject_id)
d$node_idx   <- as.factor(d$node_idx)

# Exclude same datasets as before
d <- droplevels(subset(d, !(dataset %in% c())))

cat(sprintf("Rows: %s | Subjects: %d | Nodes: %d\n",
            format(nrow(d), big.mark=","),
            nlevels(d$subject_id),
            nlevels(d$node_idx)))
cat(sprintf("Age: %.1f–%.1f | Ratio: mean=%.3f SD=%.3f >1=%.0f%%\n",
            min(d$age), max(d$age),
            mean(d$Ratio, na.rm=TRUE), sd(d$Ratio, na.rm=TRUE),
            100*mean(d$Ratio > 1, na.rm=TRUE)))

# ================================================================
# MODEL 1: Global age effect on nodal Ratio
# s(age):       shared non-linear age curve across all nodes
# s(strength):  control for nodal strength (local covariate)
# s(subject_id, bs="re"): random intercept per subject
#               — absorbs between-subject mean differences
#               — makes node-level observations within a subject valid
# dataset:      fixed site effect
# ================================================================
cat("\nFitting Model 1: Ratio ~ s(age) + s(strength) + s(subject_id,re) + sex + dataset\n")
cat("(this may take a few minutes on large N...)\n")

m1 <- bam(                               # bam() = fast GAM for large N
  Ratio ~ s(age, bs="cr", k=8) +
    s(strength, bs="cr", k=5) +
    s(subject_id, bs="re") +             # random intercept per subject
    sex + dataset,
  data=d, method="fREML",                # fREML = fast REML for bam
  discrete=TRUE                          # further speed-up
)
sm1 <- summary(m1)
cat(sprintf("\nModel 1: R²=%.4f  Dev.expl=%.2f%%  AIC=%.1f\n",
            sm1$r.sq, sm1$dev.expl*100, AIC(m1)))
print(sm1$s.table)
print(sm1$p.table)

# Partial r (age ~ Ratio after subject + dataset + sex + strength)
m_null1 <- bam(Ratio ~ s(strength, bs="cr", k=5) +
                 s(subject_id, bs="re") + sex + dataset,
               data=d, method="fREML", discrete=TRUE)
r_part1 <- cor.test(d$age, residuals(m_null1))
cat(sprintf("\nPartial r(age, Ratio | all covariates) = %.4f  p=%.3e\n",
            r_part1$estimate, r_part1$p.value))

# ================================================================
# MODEL 2: Add node-specific random smooth (per-node age trajectory)
# s(age, node_idx, bs="fs"): each node allowed its own age curve
# This tests whether the age effect varies across brain regions
# ================================================================
cat("\nFitting Model 2: + s(age, node_idx, bs='fs') — per-node trajectory\n")
suppressWarnings(
  m2 <- bam(
    Ratio ~ s(age, bs="cr", k=6) +
      s(age, node_idx, bs="fs", k=4) +   # random smooth per node
      s(strength, bs="cr", k=5) +
      s(subject_id, bs="re") +
      sex + dataset,
    data=d, method="fREML", discrete=TRUE
  )
)
sm2 <- summary(m2)
cat(sprintf("Model 2: R²=%.4f  Dev.expl=%.2f%%  AIC=%.1f  ΔAIC=%.1f vs M1\n",
            sm2$r.sq, sm2$dev.expl*100, AIC(m2), AIC(m2)-AIC(m1)))
print(sm2$s.table)

# ================================================================
# MODEL 3: Y_obs ~ s(age) + Y_null + ... (Option B at nodal level)
# If slope of Y_null ≈ 1 and age term vanishes → ratio flat by construction
# ================================================================
cat("\nFitting Model 3 (Option B): Y_obs ~ s(age) + Y_null + ...\n")
m3 <- bam(
  Y_obs ~ s(age, bs="cr", k=8) +
    Y_null +
    s(strength, bs="cr", k=5) +
    s(subject_id, bs="re") +
    sex + dataset,
  data=d, method="fREML", discrete=TRUE
)
sm3 <- summary(m3)
cat(sprintf("Model 3: R²=%.4f  Dev.expl=%.2f%%  AIC=%.1f\n",
            sm3$r.sq, sm3$dev.expl*100, AIC(m3)))
print(sm3$s.table)
print(sm3$p.table)
cat(sprintf("Y_null coefficient: β=%.4f  t=%.2f  p=%.3e\n",
            sm3$p.table["Y_null","Estimate"],
            sm3$p.table["Y_null","t value"],
            sm3$p.table["Y_null","Pr(>|t|)"]))

# ================================================================
# SUMMARY TABLE
# ================================================================
age_row <- function(sm) {
  r <- sm$s.table[grepl("^s\\(age\\)$", rownames(sm$s.table)), , drop=FALSE]
  if (nrow(r)==0) return(c(NA,NA,NA))
  c(round(r[1,"edf"],3), round(r[1,"F"],3), signif(r[1,"p-value"],4))
}
cmp <- data.frame(
  Model     = c("M1: Ratio ~ s(age)+s(str)+re(subj)+sex+ds",
                "M2: M1 + s(age,node,fs)",
                "M3: Y_obs ~ s(age)+Y_null+s(str)+re(subj)"),
  AIC       = round(c(AIC(m1), AIC(m2), AIC(m3)), 1),
  R2        = round(c(sm1$r.sq, sm2$r.sq, sm3$r.sq), 4),
  DevExpl   = round(c(sm1$dev.expl, sm2$dev.expl, sm3$dev.expl)*100, 2),
  age_edf   = c(age_row(sm1)[1], age_row(sm2)[1], age_row(sm3)[1]),
  age_F     = c(age_row(sm1)[2], age_row(sm2)[2], age_row(sm3)[2]),
  age_p     = c(age_row(sm1)[3], age_row(sm2)[3], age_row(sm3)[3]),
  partial_r = c(round(r_part1$estimate,4), NA, NA)
)
cat("\n=== MODEL COMPARISON ===\n")
print(cmp)
write.csv(cmp, file.path(out,"model_comparison.csv"), row.names=FALSE)

# ================================================================
# PLOTS — base R only (avoids memory issues with large N scatter)
# ================================================================

# Age smooth from M1
pdf(file.path(out,"01_age_smooth_M1.pdf"), width=9, height=5)
par(mar=c(4.5,4.5,3.5,1))
plot(m1, select=1, shade=TRUE, shade.col="lightblue",
     seWithMean=TRUE,
     xlab="Age (years)", ylab="Partial effect on Ratio",
     main=sprintf("Nodal Ratio ~ s(age)  [edf=%.2f  F=%.2f  p=%.4f]\nN=%s node-observations, random intercept per subject",
                  age_row(sm1)[1], age_row(sm1)[2], age_row(sm1)[3],
                  format(nrow(d), big.mark=",")),
     cex.main=0.95, cex.lab=1.1)
abline(h=0, lty=2, col="grey50")
dev.off()

# Strength smooth from M1
pdf(file.path(out,"02_strength_smooth_M1.pdf"), width=9, height=5)
par(mar=c(4.5,4.5,3.5,1))
plot(m1, select=2, shade=TRUE, shade.col="lightgreen",
     seWithMean=TRUE,
     xlab="Nodal strength", ylab="Partial effect on Ratio",
     main="Strength covariate effect on Ratio",
     cex.lab=1.1)
abline(h=0, lty=2, col="grey50")
dev.off()

# Both smooths from M3 (Y_obs)
pdf(file.path(out,"03_M3_age_smooth.pdf"), width=9, height=5)
par(mar=c(4.5,4.5,3.5,1))
plot(m3, select=1, shade=TRUE, shade.col="#FADBD8",
     seWithMean=TRUE,
     xlab="Age (years)", ylab="Partial effect on Y_obs",
     main=sprintf("Option B nodal: Y_obs ~ s(age) + Y_null\nAge smooth edf=%.2f  F=%.2f  p=%.4f",
                  age_row(sm3)[1], age_row(sm3)[2], age_row(sm3)[3]),
     cex.main=0.95, cex.lab=1.1)
abline(h=0, lty=2, col="grey50")
dev.off()

cat(sprintf("\n=== DONE ===\nOutputs: %s\n", out))
cat(sprintf("Key result: age smooth on Ratio  edf=%.2f  F=%.2f  p=%.4f\n",
            age_row(sm1)[1], age_row(sm1)[2], as.numeric(age_row(sm1)[3])))
cat(sprintf("Partial r(age|covariates) = %.4f  p=%.3e\n",
            r_part1$estimate, r_part1$p.value))

