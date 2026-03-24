# ================================================================
# AGE-STRATIFIED HARMONIZATION — fixed + dataset 6 excluded
# Key findings from first run:
#   - Dataset 6 (ages ~5-25 only) cannot be harmonized with adult
#     datasets 7,8,9 because it shares anchor ages only with other
#     young-only datasets (3,4,5) → excluded here
#   - After exclusion: 1/7 → expected 0/6 significant
#   - GAM smooth p=0.0036, edf=4.57 → real non-linear (U-shaped) effect
#   - Partial r≈0 because U-shape cancels in linear correlation
# ================================================================
library(mgcv); library(visreg); library(ggplot2); library(dplyr)
library(limma)

out <- "C:/Users/Ale/Desktop/UB/NetLearn/procdata/Rcombat_harm/age_stratified/"
dir.create(out, showWarnings=FALSE, recursive=TRUE)
gg <- function(f, p, w=11, h=5) ggsave(file.path(out, f), p, width=w, height=h)

d_all <- read.csv("C:/Users/Ale/Desktop/UB/NetLearn/procdata/demographics_with_disparity_dev_unthresholded.csv")
d_all$sex <- as.factor(d_all$sex); d_all$dataset <- as.factor(d_all$dataset)

EXCL_DS <- c(1, 2, 6)   # 1,2: scanner artefacts; 6: no adult anchor overlap
d_all   <- droplevels(subset(d_all, !(dataset %in% EXCL_DS)))
METRIC  <- "Disparity_Ratio"
BIN_W   <- 5

cat(sprintf("N=%d | Datasets: %s | Age: %.1f-%.1f\n",
            nrow(d_all), paste(levels(d_all$dataset), collapse=","),
            min(d_all$age), max(d_all$age)))

d_all$age_bin <- floor(d_all$age / BIN_W) * BIN_W
n_ds_per_bin  <- d_all %>%
  group_by(age_bin) %>%
  summarise(n_datasets=n_distinct(dataset),
            datasets=paste(sort(unique(dataset)), collapse=","), n=n(), .groups="drop")
anchor_bins <- n_ds_per_bin$age_bin[n_ds_per_bin$n_datasets >= 2]
cat(sprintf("Anchor bins: %d / %d | Anchor subjects: %d / %d (%.0f%%)\n",
            length(anchor_bins), nrow(n_ds_per_bin),
            sum(d_all$age_bin %in% anchor_bins), nrow(d_all),
            100*mean(d_all$age_bin %in% anchor_bins)))
print(n_ds_per_bin)

gg("00_coverage.pdf",
   ggplot(d_all, aes(x=age, y=dataset, colour=dataset)) +
     geom_jitter(height=0.25, alpha=0.35, size=0.8) +
     geom_vline(xintercept=anchor_bins, alpha=0.15, colour="green4") +
     labs(title="Dataset x Age coverage (excl. ds 1,2,6)",
          subtitle="Green = anchor bins (>=2 datasets co-present)",
          x="Age (years)", y="Dataset") +
     theme_bw(base_size=12)+theme(legend.position="none",
                                  plot.title=element_text(face="bold")), w=12, h=4)

# ── APPROACH A: Anchor-age offset ────────────────────────────────
d_anchor <- d_all[d_all$age_bin %in% anchor_bins, ]
m_anch   <- gam(as.formula(sprintf("%s ~ s(age,bs='cr',k=6) + sex", METRIC)),
                data=d_anchor, method="REML")
d_anchor$resid <- residuals(m_anch, type="response")
ds_off   <- d_anchor %>% group_by(dataset) %>%
  summarise(offset_A=mean(resid,na.rm=TRUE), n=n(), .groups="drop")
cat("Offsets A:\n"); print(ds_off)

d_all <- d_all %>% left_join(ds_off[,c("dataset","offset_A")], by="dataset") %>%
  mutate(!!paste0(METRIC,"_A") := !!sym(METRIC) - offset_A)

m_hA  <- gam(as.formula(sprintf("%s_A ~ s(age,by=dataset,bs='cr',k=4)+sex+dataset", METRIC)),
             data=d_all, method="REML")
age_A <- summary(m_hA)$s.table[grepl("s\\(age\\)",rownames(summary(m_hA)$s.table)),]
n_A   <- sum(age_A[,"p-value"] < 0.05)
cat(sprintf("Harmonization A: %d/%d sig\n", n_A, nrow(age_A)))
print(round(age_A[,c("edf","F","p-value")], 4))

# ── APPROACH B: Factor-smooth GAM ────────────────────────────────
suppressWarnings(
  m_fs <- gam(as.formula(sprintf(
    "%s ~ s(age,bs='cr',k=6)+s(age,dataset,bs='fs',k=4)+sex", METRIC)),
    data=d_all, method="REML"))
modal_ds  <- names(which.max(table(d_all$dataset)))
nd_global <- d_all; nd_global$dataset <- factor(modal_ds, levels=levels(d_all$dataset))
d_all[[paste0(METRIC,"_B")]] <- predict(m_fs,newdata=nd_global) + residuals(m_fs,type="response")

m_hB  <- gam(as.formula(sprintf("%s_B ~ s(age,by=dataset,bs='cr',k=4)+sex+dataset", METRIC)),
             data=d_all, method="REML")
age_B <- summary(m_hB)$s.table[grepl("s\\(age\\)",rownames(summary(m_hB)$s.table)),]
n_B   <- sum(age_B[,"p-value"] < 0.05)
cat(sprintf("Harmonization B: %d/%d sig\n", n_B, nrow(age_B)))

# ── APPROACH C: Age-bin limma ─────────────────────────────────────
d_all[[paste0(METRIC,"_C")]] <- d_all[[METRIC]]
for (bin in anchor_bins) {
  idx <- which(d_all$age_bin == bin)
  d_b <- d_all[idx,]
  if (n_distinct(d_b$dataset) < 2 || nrow(d_b) < 10) next
  des <- model.matrix(~ as.numeric(as.character(d_b$sex)) + d_b$age)
  tryCatch({
    h <- removeBatchEffect(matrix(d_b[[METRIC]],nrow=1),
                           batch=as.character(d_b$dataset), design=des)
    d_all[[paste0(METRIC,"_C")]][idx] <- as.numeric(h[1,])
  }, error=function(e) NULL)
}
m_hC  <- gam(as.formula(sprintf("%s_C ~ s(age,by=dataset,bs='cr',k=4)+sex+dataset", METRIC)),
             data=d_all, method="REML")
age_C <- summary(m_hC)$s.table[grepl("s\\(age\\)",rownames(summary(m_hC)$s.table)),]
n_C   <- sum(age_C[,"p-value"] < 0.05)
cat(sprintf("Harmonization C: %d/%d sig\n", n_C, nrow(age_C)))

# ── Best approach ─────────────────────────────────────────────────
best       <- which.min(c(n_A, n_B, n_C))
best_label <- c("A","B","C")[best]
best_col   <- paste0(METRIC,"_",best_label)
cat(sprintf("Best: Approach %s (%d/%d sig)\n", best_label, c(n_A,n_B,n_C)[best], nrow(age_A)))

m_final <- gam(as.formula(sprintf("%s ~ s(age,bs='cr',k=8)+sex+dataset", best_col)),
               data=d_all, method="REML")
sf <- summary(m_final)
cat(sprintf("Final: R2=%.4f  Dev=%.2f%%  AIC=%.1f\n", sf$r.sq, sf$dev.expl*100, AIC(m_final)))
print(sf$s.table); print(sf$p.table)

m_null <- gam(as.formula(sprintf("%s ~ sex+dataset", best_col)), data=d_all, method="REML")
r_lin  <- cor.test(d_all$age, residuals(m_null))
d_all$age2 <- d_all$age^2
m_quad <- lm(residuals(m_null) ~ d_all$age + d_all$age2)
cat(sprintf("Linear partial r=%.4f (p=%.3e)\n", r_lin$estimate, r_lin$p.value))
cat(sprintf("Quadratic term: p=%.4f %s\n", summary(m_quad)$coef[3,4],
            ifelse(summary(m_quad)$coef[3,4]<0.05,"-> U-shape confirmed","-> linear")))

m_F <- gam(as.formula(sprintf("%s ~ s(age,bs='cr',k=8)+dataset",best_col)),
           data=droplevels(subset(d_all,sex==1)), method="REML")
m_M <- gam(as.formula(sprintf("%s ~ s(age,bs='cr',k=8)+dataset",best_col)),
           data=droplevels(subset(d_all,sex==0)), method="REML")
cat("Female:\n"); print(summary(m_F)$s.table)
cat("Male:\n");   print(summary(m_M)$s.table)

# ── Predictions ───────────────────────────────────────────────────
age_seq  <- seq(floor(min(d_all$age)), ceiling(max(d_all$age)), by=0.5)
new_pred <- data.frame(age=age_seq,
                       sex    =factor(levels(d_all$sex)[1],   levels=levels(d_all$sex)),
                       dataset=factor(modal_ds,               levels=levels(d_all$dataset)))
m_raw    <- gam(as.formula(sprintf("%s~s(age,bs='cr',k=8)+sex+dataset",METRIC)),
                data=d_all, method="REML")
pr_raw   <- predict(m_raw,   newdata=new_pred, se.fit=TRUE)
pr_final <- predict(m_final, newdata=new_pred, se.fit=TRUE)

# FIX: pre-compute legend labels as plain strings (no sprintf in named-vector key)
lbl_raw  <- "Raw (no harmonization)"
lbl_best <- paste0("Approach ", best_label, " (", c(n_A,n_B,n_C)[best], "/", nrow(age_A), " sig)")
col_map  <- c("grey50","#2980B9"); names(col_map) <- c(lbl_raw, lbl_best)

traj_comp <- rbind(
  data.frame(age=age_seq, fit=pr_raw$fit,
             lo=pr_raw$fit-2*pr_raw$se.fit, hi=pr_raw$fit+2*pr_raw$se.fit, version=lbl_raw),
  data.frame(age=age_seq, fit=pr_final$fit,
             lo=pr_final$fit-2*pr_final$se.fit, hi=pr_final$fit+2*pr_final$se.fit, version=lbl_best))
traj_comp$version <- factor(traj_comp$version, levels=c(lbl_raw,lbl_best))

gg("01_raw_vs_harmonized.pdf",
   ggplot(traj_comp, aes(x=age,y=fit,colour=version,fill=version)) +
     geom_ribbon(aes(ymin=lo,ymax=hi),alpha=0.15,colour=NA) +
     geom_line(linewidth=1.3) +
     scale_colour_manual(values=col_map) + scale_fill_manual(values=col_map) +
     labs(title="Age trajectory: raw vs age-stratified harmonization",
          subtitle="Similar shape = signal preserved | Large divergence = over-correction",
          x="Age (years)", y=METRIC, colour="", fill="") +
     theme_bw(base_size=12)+theme(legend.position="bottom",
                                  plot.title=element_text(face="bold")), w=12, h=6)

# ── Harmonization heatmap ─────────────────────────────────────────
make_hdf <- function(tab, lbl) {
  df <- as.data.frame(tab[,c("edf","F","p-value")])
  df$dataset  <- sub("s\\(age\\):dataset","", rownames(tab))
  df$approach <- lbl; df
}
harm_df <- rbind(make_hdf(age_A,"A: Anchor"),make_hdf(age_B,"B: Factor-smooth"),make_hdf(age_C,"C: Age-bin"))
gg("02_heatmap.pdf",
   ggplot(harm_df, aes(x=dataset,y=approach,fill=-log10(`p-value`))) +
     geom_tile(colour="white",linewidth=0.5) +
     geom_text(aes(label=sprintf("p=%.3f",`p-value`)),size=3.2) +
     scale_fill_gradient(low="white",high="#E74C3C",name="-log10(p)") +
     labs(title="Harmonization quality (excl. ds 1,2,6)",
          subtitle="White=good (p>=0.05) | Red=residual site effect",
          x="Dataset", y="Approach") +
     theme_bw(base_size=11)+theme(plot.title=element_text(face="bold")), w=10, h=4)

# ── Base R plots ──────────────────────────────────────────────────
suppressWarnings({
  pdf(file.path(out,"03_final_smooth.pdf"), width=9, height=5)
  par(mar=c(4.5,4.5,3.5,1))
  plot(m_final, select=1, shade=TRUE, shade.col="lightblue", seWithMean=TRUE,
       xlab="Age (years)", ylab=paste("Partial effect on",best_col),
       main=sprintf("Approach %s | edf=%.2f F=%.2f p=%.4f | partial r=%.3f p=%.3f",
                    best_label, sf$s.table[1,"edf"], sf$s.table[1,"F"],
                    sf$s.table[1,"p-value"], r_lin$estimate, r_lin$p.value),
       cex.main=0.82, cex.lab=1.1)
  abline(h=0, lty=2, col="grey50")
  dev.off()
  
  pdf(file.path(out,"04_sex_smooths.pdf"), width=11, height=5)
  par(mfrow=c(1,2), mar=c(4.5,4.5,3,1), oma=c(0,0,3,0))
  plot(m_F,select=1,shade=TRUE,shade.col="#FADBD8",seWithMean=TRUE,
       xlab="Age",ylab="Partial effect",main="Female",cex.lab=1.1); abline(h=0,lty=2,col="grey50")
  plot(m_M,select=1,shade=TRUE,shade.col="#D6EAF8",seWithMean=TRUE,
       xlab="Age",ylab="Partial effect",main="Male",  cex.lab=1.1); abline(h=0,lty=2,col="grey50")
  mtext(paste(best_col,"— by sex"), outer=TRUE, cex=1.1, font=2)
  dev.off()
})

# ── Save ──────────────────────────────────────────────────────────
v <- visreg(m_final, "age", plot=FALSE)
write.csv(data.frame(age=v$res$age, ratio_res=v$res$visregRes,
                     dataset=d_all$dataset, sex=d_all$sex),
          file.path(out,"residuals_harmonized.csv"), row.names=FALSE)
write.csv(d_all, file.path(out,"data_harmonized.csv"), row.names=FALSE)

cat(sprintf("\n=== DONE === Outputs: %s\n", out))
cat(sprintf("Age smooth: edf=%.2f F=%.2f p=%.4f\n",
            sf$s.table[1,"edf"], sf$s.table[1,"F"], sf$s.table[1,"p-value"]))

