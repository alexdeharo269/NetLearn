library(mgcv)
library(visreg)
# 2. Load data containing Dev from null model of Disparity
data_dev <- read.csv("C:/Users/Ale/Desktop/UB/NetLearn/procdata/demographics_with_disparity_dev.csv")

data_dev$sex <- as.factor(data_dev$sex)
data_dev$dataset <- as.factor(data_dev$dataset)
data_dev$atlas <- as.factor(data_dev$atlas)

model_disp_dev <- gam(Disparity_Deviation ~ s(age, bs="cr") + sex + dataset + atlas, 
                      data=data_dev, method='REML')

print(summary(model_disp_dev))


v_disp_dev <- visreg(model_disp_dev, "age", plot=FALSE)


export_df <- data.frame(age = v_disp_dev$res$age,
disparity_res_dev=v_disp_dev$res$visregRes)

write.csv(export_df, "C:/Users/Ale/Desktop/UB/NetLearn/procdata/disparity_residuals_dev.csv", row.names=FALSE)

