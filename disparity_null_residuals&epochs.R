
library(mgcv)
library(visreg)
library(gratia)

# 1. Load and format data
data_dev <- read.csv("C:/Users/Ale/Desktop/UB/NetLearn/procdata/demographics_with_disparity_dev.csv")
data_dev$sex <- as.factor(data_dev$sex)
data_dev$dataset <- as.factor(data_dev$dataset)
data_dev$atlas <- as.factor(data_dev$atlas)

# 2. Fit the model (Corrected to Disparity_Deviation + gamma penalty for large N)
model_disp_dev <- gam(Disparity_Deviation ~ s(age, bs="cr", k=10) + sex + dataset + atlas, 
                      data=data_dev, method='REML', gamma=1.0)

# 3. Export Partial Residuals
v_disp_dev <- visreg(model_disp_dev, "age", plot=FALSE)
export_df <- data.frame(age = v_disp_dev$res$age, disparity_res_dev = v_disp_dev$res$visregRes)
write.csv(export_df, "C:/Users/Ale/Desktop/UB/NetLearn/procdata/disparity_residuals_dev.csv", row.names=FALSE)

# 4. Export the Exact GAM Spline 
smooth_fit <- smooth_estimates(model_disp_dev, select = "s(age)")
write.csv(data.frame(age = smooth_fit$age, fit = smooth_fit$.estimate), 
          "C:/Users/Ale/Desktop/UB/NetLearn/procdata/gam_smooth_curve.csv", row.names=FALSE)

# 5. Calculate Derivatives and Extract Exact Epoch Turning Points
deriv <- derivatives(model_disp_dev, select = "s(age)")
sign_changes <- which(diff(sign(deriv$.derivative)) != 0)
roots <- deriv$age[sign_changes]

cat("\n=== EXACT GAM TURNING POINTS (EPOCHS) ===\n")
print(roots)

# Export the turning points to Python
write.csv(data.frame(turning_points = roots), 
          "C:/Users/Ale/Desktop/UB/NetLearn/procdata/turning_points.csv", row.names=FALSE)

