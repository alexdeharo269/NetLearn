library(mgcv)
library(visreg)

# 1. Load and factorize data
data <- read.csv("C:/Users/Ale/Desktop/UB/NetLearn/procdata/demographics_with_graph_metrics.csv")
data$sex <- as.factor(data$sex)
data$dataset <- as.factor(data$dataset)
data$atlas <- as.factor(data$atlas)

# 2. Harmonization Check (Corrected to include atlas)
# Forces macro-trajectory (k=4) and isolates Density to compare pure age trends across scanners
model_check_rigorous <- gam(Disparity_Y ~ s(age, by=dataset, bs="cr", k=4) + 
                              sex + dataset + atlas + 
                              s(Density, bs="cr", k=5), 
                            data=data, method='REML')

plot(model_check_rigorous, shade=TRUE, )

# 3. Main Models (Atlas included)
model_disp <- gam(Disparity_Y ~ s(age, bs="cr") + sex + dataset + atlas, 
                  data=data, method='REML')

model_disp_density <- gam(Disparity_Y ~ s(age, bs="cr") + sex + dataset + atlas +
                            s(Density, bs="cr", k=5),
                          data=data, method='REML')

# 4. Sex-Specific Models (Corrected to include Density and Atlas)
F_data <- subset(data, sex == 1)
M_data <- subset(data, sex == 0)

F_model <- gam(Disparity_Y ~ s(age, bs="cr") + dataset + atlas + s(Density, bs="cr", k=5), 
               data=F_data, method='REML')
M_model <- gam(Disparity_Y ~ s(age, bs="cr") + dataset + atlas + s(Density, bs="cr", k=5), 
               data=M_data, method='REML')

print(summary(model_disp_density))

# 5. Extract Partial Residuals
v_disp <- visreg(model_disp, "age", plot=FALSE)
v_disp_den <- visreg(model_disp_density, "age", plot=FALSE)
F_vis <- visreg(F_model, "age", plot=FALSE)
M_vis <- visreg(M_model, "age", plot=FALSE)

# 6. Export DataFrames
export_df <- data.frame(
  age = v_disp$res$age,
  disparity_res = v_disp$res$visregRes,
  disparity_res_sden = v_disp_den$res$visregRes
)

export_dfsex <- rbind(
  data.frame(age = M_vis$res$age, disparity_res = M_vis$res$visregRes, Sex = "Male"),
  data.frame(age = F_vis$res$age, disparity_res = F_vis$res$visregRes, Sex = "Female")
)

write.csv(export_df, "C:/Users/Ale/Desktop/UB/NetLearn/procdata/disparity_residuals.csv", row.names=FALSE)
write.csv(export_dfsex, "C:/Users/Ale/Desktop/UB/NetLearn/procdata/disparity_residuals_sex.csv", row.names=FALSE)