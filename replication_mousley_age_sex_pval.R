library(mgcv)
library(visreg)
library(dplyr)
library(R.matlab)

# Rutas
input_mat_path <- "C:/Users/Ale/Desktop/UB/NetLearn/procdata/density-controlled_organizational_measures_alexa.mat"
input_path <- "C:/Users/Ale/Desktop/UB/NetLearn/procdata/demographics_with_graph_metrics_harm.csv"

demo_path <- "C:/Users/Ale/Desktop/UB/NetLearn/preprocdata/demographics_harm.csv"

# Cargar y alinear datos
demographics <- read.csv(demo_path)
measures <- readMat(input_mat_path)$organizational.measures

# 2. Replicar la estructura EXACTA de la autora
global_data <- as.data.frame(cbind(demographics, measures))

data_ours <- read.csv(input_path)
global_data_ours <- as.data.frame(data_ours)


# 3. Asignar nombres EXACTOS como ella hace
colnames(global_data) <- c('age','sex','dataset','atlas','remove_index',
                           'global_efficiency','path_length','small_worldness','strength',
                           'modularity','core_periphery','kcore','score',
                           'local_efficiency','clustering','betweenness','subgraph_centrality')

# 4. El GAM que da 0.543
# IMPORTANTE: En su script NO usa as.factor(). Los trata como numéricos.
model <- gam(strength ~ s(age, bs="cr") + sex + dataset + atlas, 
             data=global_data, method='REML')

summary(model)

model_ours <- gam(Strength_CD ~ s(age, bs="cr") + sex + dataset + atlas, 
             data=global_data_ours, method='REML')

summary(model_ours)


