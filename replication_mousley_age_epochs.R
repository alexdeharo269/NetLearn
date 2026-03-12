library(mgcv)
library(dplyr)
library(R.matlab)
library(visreg)
library(ggplot2)

# 1. Cargar datos
demo_path <- "C:/Users/Ale/Desktop/UB/NetLearn/preprocdata/demographics_harm.csv"
mat_path <- "C:/Users/Ale/Desktop/UB/NetLearn/procdata/density-controlled_organizational_measures_alexa.mat"

demographics <- read.csv(demo_path)
# Eliminar 'remove_index' para que queden exactamente 4 columnas
demographics <- demographics[, c("age", "sex", "dataset", "atlas")]

measures_mat <- readMat(mat_path)
measures <- measures_mat$organizational.measures

# 2. Replicar estructura exacta de la autora
global_data <- as.data.frame(bind_cols(demographics, as.data.frame(measures)))
col_names <- c('age','sex','dataset','atlas',
               'global_efficiency','path_length','small_worldness','strength','modularity',
               'core_periphery','kcore','score','local_efficiency','clustering','betweenness','subgraph_centrality')
colnames(global_data) <- col_names

# 3. Ajustar el GAM (Sin usar as.factor para obtener el 0.543 exacto)
strength_model <- gam(strength ~ s(age, bs="cr") + sex + dataset + atlas,
                      data=global_data, method='REML')

# Imprimir el p-valor de sex
sum_mod <- summary(strength_model)
p_sex <- sum_mod$p.table["sex", "Pr(>|t|)"]
cat(sprintf("\nP-valor Sexo (Strength): %.3f\n\n", p_sex))

# 4. Extraer los residuos parciales EXACTAMENTE como la autora
# Al no poner "age", visreg devuelve una lista y el índice [[1]] funciona
reg_plots <- visreg(strength_model, gg=TRUE, type="conditional")

export_df <- data.frame(
  age = reg_plots[[1]]$data$x,
  strength = reg_plots[[1]]$data$y
)

# Exportar
write.csv(export_df, "C:/Users/Ale/Desktop/UB/NetLearn/procdata/strength_residuals.csv", row.names=FALSE)

