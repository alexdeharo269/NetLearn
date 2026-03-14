library(mgcv)
library(visreg)

# 1. Load data containing Disparity_Y (calculated from Variable Density connectomes)
data <- read.csv("C:/Users/Ale/Desktop/UB/NetLearn/procdata/demographics_filtered_with_graph_metrics.csv")




# 2. Factorize categorical confounders (Correct statistical approach)
data$sex <- as.factor(data$sex)
data$dataset <- as.factor(data$dataset)
data$atlas <- as.factor(data$atlas)



# 3. Fit GAM
model_disp <- gam(Disparity_Y ~ s(age, bs="cr") + sex + dataset, 
                  data=data, method='REML')

# Check harmonization across datasets

# Fit a model where each dataset gets its own aging curve
model_check <- gam(Disparity_Y ~ s(age, by=dataset, bs="cr") + sex + dataset + atlas, 
                   data=data, method='REML')

# Plot all curves together
plot(model_check, pages=1, shade=TRUE)



#Robustness check including density as non-linear spline.
model_disp_density <- gam(Disparity_Y ~ s(age, bs="cr") + sex + dataset +
                            s(Density, bs="cr", k=5),
                          data=data, method='REML')

F_data <- subset(data, sex == 1)
M_data <- subset(data, sex == 0)

# 2. Fit separate models
F_model <- gam(Disparity_Y ~ s(age, bs="cr") + dataset, data=F_data, method='REML')
M_model <- gam(Disparity_Y ~ s(age, bs="cr") + dataset, data=M_data, method='REML')


# Print summary to check significance
print(summary(model_disp))
print(summary(model_disp_density))


# 4. Extract partial residuals via visreg
v_disp <- visreg(model_disp, "age", plot=FALSE)
v_disp_den <- visreg(model_disp_density, "age", plot=FALSE)

F_vis <- visreg(F_model, "age", plot=FALSE)
M_vis <- visreg(M_model, "age", plot=FALSE)
export_df <- data.frame(
  age = v_disp$res$age,
  disparity_res = v_disp$res$visregRes,
  disparity_res_sden=v_disp_den$res$visregRes
)

df_M <- data.frame(
  age = M_vis$res$age,
  disparity_res = M_vis$res$visregRes,
  Sex = "Male")
df_F <- data.frame(
  age = F_vis$res$age,
  disparity_res = F_vis$res$visregRes,
  Sex = "Female")
export_dfsex <- rbind(df_M, df_F)

write.csv(export_df, "C:/Users/Ale/Desktop/UB/NetLearn/procdata/disparity_residuals.csv", row.names=FALSE)

write.csv(export_dfsex, "C:/Users/Ale/Desktop/UB/NetLearn/procdata/disparity_residuals_sex.csv", row.names = FALSE)
