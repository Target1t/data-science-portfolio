library(tidyverse)
library(rpart)
library(rpart.plot)
library(pROC)
library(caret)
library(psych)
library(knitr)
library(e1071)
library(ggplot2)
library(writexl)
library(dplyr)
library(ggpubr)
library(reshape2)
library(RColorBrewer)

# 四、描述性统计

# 0.数据清洗
## 缺失值检测
wine <- read.csv("C:\\Users\\86178\\Desktop\\Tongruan\\WineQT.csv")
any(is.na(wine))
## 构建二元变量
wine$binary_quality <- ifelse(wine$quality > 5, 1, 0)
table(wine$binary_quality)


# 1.总体描述
wine <- wine %>% select(-Id, -quality)

# 计算描述性统计
desc_stats <- describe(wine)
desc_stats$skewness <- apply(wine, 2, skewness)
desc_stats$kurtosis <- apply(wine, 2, kurtosis)

# 保留关键列
final_table <- desc_stats[, c("mean", "sd", "min", "max", "skewness", "kurtosis", "n")]
colnames(final_table) <- c("均值", "标准差", "最小值", "最大值", "偏度", "峰度", "样本数")

# 绘制三线统计描述表
kable(final_table, caption = "红酒数据的三线统计描述表", digits = 2)
write_xlsx(final_table, path = "统计表.xlsx")


# 统计频数与百分比
binary_quality_freq <- wine %>%
  count(binary_quality) %>%
  mutate(
    label_text = ifelse(binary_quality == 1, "好", "一般"),
    percent = round(n / sum(n) * 100, 1),
    label = paste0(label_text, "\n", n, " 个\n", percent, "%")
  )

# 1.1 绘制质量饼图 
### AI参与：为了实现更好看的效果，ggplot部分参数（position，theme——void等）由 AI 生成，下同
ggplot(binary_quality_freq, aes(x = "", y = n, fill = factor(label_text))) +
  geom_bar(width = 1, stat = "identity", color = "white") +
  coord_polar(theta = "y") +
  geom_text(aes(label = label), position = position_stack(vjust = 0.5), color = "white", size = 5) +
  labs(title = "binary_quality 质量饼状图", fill = "质量分类") +
  theme_void() +
  scale_fill_manual(values = c("#FF6F61", "#6A5ACD"))  

# 1.2 酸度
ggplot(wine, aes(x = pH)) + 
  geom_histogram(bins = 30, fill = "#FFFACD", color = "black") +
  labs(title = "pH Distribution")

# 选择相关酸度变量
acid_vars <- wine %>% 
  select(fixed.acidity, volatile.acidity, citric.acid) %>%
  pivot_longer(cols = everything(), names_to = "Acid_Type", values_to = "Value")

ggplot(acid_vars, aes(x = Value, fill = Acid_Type)) +
  geom_histogram(bins = 30, color = "black") +
  facet_wrap(~ Acid_Type, scales = "free", ncol = 1) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "三种酸度成分的分布", x = "值", y = "频数") +
  theme_minimal() +
  theme(legend.position = "none")

# 1.3 无机物浓度
sulfur_data <- wine %>%
  select(chlorides, free.sulfur.dioxide, total.sulfur.dioxide, sulphates) %>%
  pivot_longer(cols = everything(), names_to = "Chemical", values_to = "Value")

ggplot(sulfur_data, aes(x = Value, fill = Chemical)) +
  geom_histogram(bins = 30, color = "black") +
  facet_wrap(~ Chemical, scales = "free", ncol = 2) +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "无机物浓度分布直方图",
       x = "取值", y = "频数") +
  theme_minimal() +
  theme(legend.position = "none")

# 1.4 剩余指标
other_data <- wine %>%
  select(residual.sugar,density,alcohol) %>%
  pivot_longer(cols = everything(), names_to = "Others", values_to = "Value")

ggplot(other_data, aes(x = Value, fill = Others)) +
  geom_histogram(bins = 30, color = "black") +
  facet_wrap(~ Others, scales = "free", ncol = 1) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "剩余指标的分布", x = "值", y = "频数") +
  theme_minimal() +
  theme(legend.position = "none")



# 2. 分类可视化
# 2.1 酸度变量
acid_vars <- wine %>%
  select(binary_quality, fixed.acidity, volatile.acidity, citric.acid, pH) %>%
  pivot_longer(cols = -binary_quality, names_to = "Acid_Type", values_to = "Value")

### AI参与：为了实现更好看的效果，geom_violin和facet_wrap部分由 AI 生成，下同
ggplot(acid_vars, aes(x = factor(binary_quality, labels = c("一般", "好")), 
                      y = Value, 
                      fill = factor(binary_quality))) +
  geom_violin(trim = FALSE, alpha = 0.4) +
  geom_boxplot(width = 0.1, outlier.shape = NA, alpha = 0.6) +
  facet_wrap(~ Acid_Type, scales = "free", ncol = 2) +
  labs(title = "不同质量红酒的酸度成分分布",
       x = "质量分类", y = "数值") +
  scale_fill_brewer(palette = "Pastel1") +
  theme_minimal() +
  theme(legend.position = "none", 
        strip.text = element_text(size = 12, face = "bold"),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5))

# 2.2 无机物数据
sulfur_vars <- wine %>%
  select(binary_quality, chlorides, free.sulfur.dioxide, total.sulfur.dioxide, sulphates) %>%
  pivot_longer(cols = -binary_quality, names_to = "Chemical", values_to = "Value")

ggplot(sulfur_vars, aes(x = factor(binary_quality, labels = c("一般", "好")), 
                        y = Value, 
                        fill = factor(binary_quality))) +
  geom_violin(trim = FALSE, alpha = 0.4) +
  geom_boxplot(width = 0.1, outlier.shape = NA, alpha = 0.6) +
  facet_wrap(~ Chemical, scales = "free", ncol = 2) +
  labs(title = "不同质量红酒的无机物浓度分布",
       x = "质量分类", y = "取值") +
  scale_fill_brewer(palette = "Set3") +
  theme_minimal() +
  theme(legend.position = "none", 
        strip.text = element_text(size = 12, face = "bold"),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5))

# 2.3 剩余数据
residual_vars <- wine %>%
  select(binary_quality, residual.sugar, density, alcohol) %>%
  pivot_longer(cols = -binary_quality, names_to = "Indicator", values_to = "Value")

ggplot(residual_vars, aes(x = factor(binary_quality, labels = c("一般", "好")),
                          y = Value, 
                          fill = factor(binary_quality))) +
  geom_violin(trim = FALSE, alpha = 0.4) +
  geom_boxplot(width = 0.1, outlier.shape = NA, alpha = 0.6) +
  facet_wrap(~ Indicator, scales = "free", ncol = 1) +
  labs(title = "不同质量红酒的剩余值指标分布",
       x = "质量分类", y = "数值") +
  scale_fill_manual(values = c("#FFA07A", "#20B2AA")) +
  theme_minimal() +
  theme(legend.position = "none",
        strip.text = element_text(size = 12, face = "bold"),
        plot.title = element_text(size = 14, face = "bold", hjust = 0.5))

# 3. 相关性分析
# 各个变量与质量的热力图
selected_vars <- wine %>%
  select(fixed.acidity, volatile.acidity, citric.acid, pH,
         chlorides, free.sulfur.dioxide, total.sulfur.dioxide, sulphates, 
         residual.sugar, density, alcohol)

# 计算皮尔逊相关系数与质量之间的相关性
cor_pearson <- sapply(selected_vars, function(x) cor(x, wine$binary_quality, method = "pearson"))

# 计算斯皮尔曼相关系数与质量之间的相关性
cor_spearman <- sapply(selected_vars, function(x) cor(x, wine$binary_quality, method = "spearman"))

cor_pearson_df <- data.frame(Variable = names(cor_pearson), Pearson = cor_pearson)
cor_spearman_df <- data.frame(Variable = names(cor_spearman), Spearman = cor_spearman)
cor_df <- merge(cor_pearson_df, cor_spearman_df, by = "Variable")


# 斯皮尔曼相关性热力图
### AI参与：我们并未系统学习热力图，该部分由 AI 生成
ggplot(cor_df, aes(x = Variable, y = "Spearman", fill = Spearman)) +
  geom_tile() +
  geom_text(aes(label = round(Spearman, 2)), color = "black", size = 5) +  
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_blank()) +
  labs(title = "各变量与质量的斯皮尔曼相关性热力图")

write_xlsx(cor_df, path = "correlation_analysis_results.xlsx")

#大图
all_vars <- wine %>%
  select(binary_quality, fixed.acidity, volatile.acidity, citric.acid, pH,
         chlorides, free.sulfur.dioxide, total.sulfur.dioxide, sulphates, 
         residual.sugar, density, alcohol)

# 将 binary_quality 转换为数值型变量
all_vars$binary_quality <- as.numeric(all_vars$binary_quality) - 1  # 将 "一般" 转为 0, "好" 转为 1

cor_matrix_all <- cor(all_vars, method = "pearson")
cor_melted_all <- melt(cor_matrix_all)

# 绘制热力图
ggplot(cor_melted_all, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1, 1), space = "Lab") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_blank()) +
  labs(title = "所有变量与质量分类的相关性热力图")

#统计推断
##分组
high_quality = subset(wine,binary_quality==1)[,c(-12,-13,-14)]
low_quality = subset(wine,binary_quality==0)[,c(-12,-13,-14)]
high_p=list()
low_p=list()


##正态性检验
for (col in colnames(high_quality)){
  high_p = append(high_p,shapiro.test(high_quality[[col]])$p.value)
  low_p = append(low_p,shapiro.test(low_quality[[col]])$p.value)
}
print(high_p)
print(low_p)


##Mann-Whitney U 检验
MWU_p=list()
for (col in colnames(high_quality)){
  MWU_p = append(MWU_p,wilcox.test(high_quality[[col]],low_quality[[col]])$p.value)
}
print(MWU_p)


##z检验
# 计算样本大小
n_low <- nrow(low_quality)
n_high <- nrow(high_quality)

# 创建结果数据框
results <- data.frame(
  Variable = character(),
  Mean_Diff = numeric(),
  CI_Lower = numeric(),
  CI_Upper = numeric(),
  Z_Value = numeric(),
  P_Value = numeric(),
  stringsAsFactors = FALSE
)

for (col in names(low_quality)) {
  # 计算组统计量
  mean_low <- mean(low_quality[[col]])
  mean_high <- mean(high_quality[[col]])
  sd_low <- sd(low_quality[[col]])
  sd_high <- sd(high_quality[[col]])
  
  # 计算均值差
  mean_diff <- mean_low - mean_high
  
  # 计算标准误
  se <- sqrt((sd_low^2/n_low) + (sd_high^2/n_high))
  
  # 计算Z值和p值
  z_value <- mean_diff / se
  p_value <- 2 * (1 - pnorm(abs(z_value)))
  
  # 计算95%置信区间
  ci_lower <- mean_diff - 1.96 * se
  ci_upper <- mean_diff + 1.96 * se
  results <- rbind(results, data.frame(
    Variable = col,
    Mean_Diff = mean_diff,
    CI_Lower = ci_lower,
    CI_Upper = ci_upper,
    Z_Value = z_value,
    P_Value = p_value
  ))
}
print(results)


# 主成分分析

library(dplyr)
library(corrplot)
library(FactoMineR)
library(factoextra)

# 0.数据清洗
## 缺失值检测
wine <- read.csv("C:\\Users\\86178\\Desktop\\Tongruan\\WineQT.csv")
any(is.na(wine))
## 构建二元变量
wine$binary_quality <- ifelse(wine$quality > 5, 1, 0)
table(wine$binary_quality)
wine <- wine %>% select(-Id, -quality)

# 1.

# 计算相关系数矩阵
cor_matrix <- cor(wine, use = "complete.obs")

# 绘制相关系数热图
### AI参与：我们并未系统学习热力图，该部分由 AI 生成
corrplot(cor_matrix,
         method = "color",        
         order = "hclust",        
         tl.col = "black",        
         tl.cex = 0.8,            
         addCoef.col = "black",   
         number.cex = 0.6,        
         col = colorRampPalette(c("blue", "white", "red"))(200))  # 渐变颜色


# 选取数值型变量用于PCA
wine_numeric <- wine[sapply(wine, is.numeric)]

# 执行主成分分析（scale.unit = TRUE 表示标准化变量）
res.pca <- PCA(wine_numeric, scale.unit = TRUE, ncp = ncol(wine_numeric), graph = FALSE)

# 查看前几个主成分的解释方差
eig.val <- res.pca$eig
print(eig.val)

# 绘制碎石图（Scree Plot）
fviz_eig(res.pca, addlabels = TRUE, ylim = c(0, 50))


# 2.1

# 获取变量相关信息
var <- get_pca_var(res.pca)

# 查看变量坐标（coord）
head(var$coord,5)

# 查看变量与主成分的相关性
head(var$cor)

# 查看变量的 cos² 值
head(var$cos2)

# 查看变量对每个主成分的贡献率
head(var$contrib)

# 可视化变量在前两个主成分上的分布图（箭头图）
fviz_pca_var(res.pca,
             col.var = "cos2", 
             gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
             repel = TRUE,     
             axes = c(1, 2))   

# 2.2 

head(var$cos2,5)

# 1. 可视化所有变量在各个主成分上的 cos² 值矩阵热图
# 提取前5个主成分的 cos² 值
cos2_top5 <- var$cos2[, 1:5]  

# 绘制热图（仅显示前5个主成分）
corrplot(cos2_top5, 
         is.corr = FALSE, 
         tl.col = "black",
         col = colorRampPalette(c("white", "blue", "red"))(200),
         mar = c(0, 0, 1, 0),
         title = "Cos² of Variables on First 5 Principal Components")

# 2. 可视化变量在前两个主成分上的 cos2（条形图形式）
# 可改 axes = c(1,2), c(1,3) 等
fviz_cos2(res.pca, choice = "var", axes = 1:2,
          top = 20, 
          col = "steelblue",
          ggtheme = theme_minimal()) +
  labs(title = "Cos2 of Variables on PC1 and PC2")

# 3. 可视化变量在因子图上的代表性（用 cos2 着色）
fviz_pca_var(res.pca, 
             col.var = "cos2", 
             gradient.cols = c("white", "blue", "red"), 
             repel = TRUE,
             axes = c(1, 2)) +
  labs(title = "Variable Factor Map Colored by Cos2")


# 2.3
# 1. 变量贡献度矩阵
var$contrib[, 1:5]  # 可查看变量在各个主成分上的贡献百分比

# 2. 可视化贡献度热图（每个主成分 × 每个变量）
### AI参与：我们并未系统学习热力图，该部分由 AI 生成
library(corrplot)
corrplot(var$contrib, is.corr = FALSE, 
         tl.col = "black", 
         col = colorRampPalette(c("white", "orange", "red"))(200),
         mar = c(0,0,1,0),
         title = "Contribution of Variables to Principal Components")

# 3. 可视化变量对 PC1 的贡献（前10个变量）
fviz_contrib(res.pca, choice = "var", axes = 1) +
  labs(title = "Top 10 Contributing Variables to PC1")

# 4. 可视化变量对 PC2 的贡献（前10个变量）
fviz_contrib(res.pca, choice = "var", axes = 2) +
  labs(title = "Top 10 Contributing Variables to PC2")

# 1. 提取变量对PC1和PC2的贡献度
contrib_dim1 <- var$contrib[, 1]
contrib_dim2 <- var$contrib[, 2]

# 2. 计算总贡献度（PC1 + PC2）
total_contrib <- contrib_dim1 + contrib_dim2

# 3. 转换为数据框用于可视化
contrib_df <- data.frame(
  Variable = names(total_contrib),
  TotalContrib = total_contrib
)

# 4. 取前10个变量进行条形图展示
library(ggplot2)
top_contrib <- contrib_df %>%
  arrange(desc(TotalContrib)) %>%
  slice(1:10)

ggplot(top_contrib, aes(x = reorder(Variable, TotalContrib), y = TotalContrib)) +
  geom_bar(stat = "identity", fill = "tomato") +
  coord_flip() +
  labs(title = "Top 10 Variable Contributions to PC1 + PC2",
       x = "Variable",
       y = "Total Contribution (%)") +
  theme_minimal()


# 可视化变量贡献：对前两个主成分的综合影响（贡献大的变量更靠近圆周）
fviz_pca_var(res.pca,
             col.var = "contrib",
             gradient.cols = c("white", "orange", "red"),
             repel = TRUE,
             axes = c(1, 2)) +
  labs(title = "Contribution of Variables to Dimensions 1 and 2")


# 获取主成分得分矩阵（每个样本在各主成分上的坐标）
# 载荷矩阵（变量在各主成分上的坐标/相关性）
loadings <- res.pca$var$coord
loadings[, 1:5]  # 查看前几行

# 获取样本在前5个主成分上的得分
scores <- res.pca$ind$coord[, 1:5]

# 获取前5个主成分的方差解释比例（第二列为%）
eig <- res.pca$eig[1:5, 2]
weights <- eig / sum(eig)  # 标准化为加权比例

# 计算综合得分（每个样本 × 权重）
composite_score <- as.matrix(scores) %*% weights

# 加入原始数据
wine$PCA_score <- composite_score

# 输出前5个红酒样本的综合得分
head(wine[, c("PCA_score")], 5)


calculate_score <- function(pca_result, original_data, quality_col = "binary_quality", top_n_dim = 5) {
  # 提取前 top_n_dim 个主成分的得分
  scores <- pca_result$ind$coord[, 1:top_n_dim]
  
  # 获取解释的方差百分比
  eig <- pca_result$eig[1:top_n_dim, 2]
  weights <- eig / sum(eig)
  
  # 计算综合得分
  composite_score <- as.matrix(scores) %*% weights
  
  # 合并结果
  result <- as.data.frame(scores)
  result$PCA_score <- composite_score
  result$Rank <- rank(-composite_score)
  
  # 添加酒品分类列
  result$Quality <- original_data[[quality_col]]
  
  return(result)
}

# 应用函数
score_df <- calculate_score(res.pca, wine, quality_col = "binary_quality", top_n_dim = 5)

# 查看前几行
head(score_df)

# 计算两类酒的平均总得分
aggregate(PCA_score ~ Quality, data = score_df, FUN = mean)

# 也可看中位数分布
aggregate(PCA_score ~ Quality, data = score_df, FUN = median)

# 可视化分布
library(ggplot2)
ggplot(score_df, aes(x = PCA_score, fill = factor(Quality))) +
  geom_density(alpha = 0.5) +
  labs(title = "PCA综合得分在不同质量红酒中的分布",
       x = "PCA综合得分", fill = "质量分类 (0 = 一般, 1 = 好)")


# 机器学习
set.seed(123)
trainIndex <- createDataPartition(wine$binary_quality, p = 0.8, list = FALSE)
train_data <- wine[trainIndex, ]
test_data <- wine[-trainIndex, ]

#用全部变量训练逻辑回归
logit_model1 <- glm(binary_quality ~ ., family = "binomial", data = train_data)
summary(logit_model1)

# 逐步回归
step_model <- step(logit_model1, direction = "both", trace = TRUE)
summary(step_model)

# 计算伪R²
null_model <- glm(binary_quality ~ 1, data = train_data, family = "binomial")
mcfadden_r2 <- 1 - logLik(step_model) / logLik(null_model)
mcfadden_r2

# 计算训练集和测试集的预测值
train_predictions_prob <- predict(step_model, train_data, type = "response")
test_predictions_prob <- predict(step_model, test_data, type = "response")
train_predictions <- ifelse(train_predictions_prob > 0.5, 1, 0)
test_predictions <- ifelse(test_predictions_prob > 0.5, 1, 0)

# 计算训练集和测试集的误差
train_data$prediction_diff <- train_predictions_prob - train_data$binary_quality
test_data$prediction_diff <- test_predictions_prob - test_data$binary_quality

# 根据不同质量分组计算偏差的均值和标准差
#该段复杂聚合计算为AI完成
train_bias_by_quality <- aggregate(prediction_diff ~ binary_quality, data = train_data, FUN = function(x) c(mean = mean(x), sd = sd(x)))
test_bias_by_quality <- aggregate(prediction_diff ~ binary_quality, data = test_data, FUN = function(x) c(mean = mean(x), sd = sd(x)))

# 打印训练集和测试集的偏差统计
print("训练集偏差统计：")
print(train_bias_by_quality)
print("测试集偏差统计：")
print(test_bias_by_quality)

# 训练集的残差
train_data$residuals <- residuals(step_model, type = "deviance")

# 对测试集进行预测
test_predictions_prob <- predict(step_model, newdata = test_data, type = "response")

# 实际值
y <- test_data$binary_quality
p <- test_predictions_prob

# 计算残差
test_data$residuals <- sign(y - p) * sqrt(-2 * (y * log(p) + (1 - y) * log(1 - p)))

#下ggplot画图代码为AI完成
ggplot(test_data, aes(x = test_predictions_prob, y = residuals, color = factor(binary_quality))) +
  geom_point(alpha = 0.6) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  ggtitle("Test Set: Deviance Residuals vs Predicted Probabilities") +
  theme_minimal()

# 树模型
#检视显著变量（从逻辑回归模型中选择）
summary(logit_model2)
# 提取显著变量用于建模
significant_vars <- c("fixed.acidity", "volatile.acidity", "citric.acid", 
                      "total.sulfur.dioxide", "sulphates", "alcohol")

#随机森林模型训练
rf_model <- randomForest(
  factor(binary_quality) ~ .,
  data = train_data[, c(significant_vars, "binary_quality")],
  importance = TRUE,
  ntree = 500
)
#可视化变量重要性
importance_df <- as.data.frame(importance(rf_model))
importance_df$Variable <- rownames(importance_df)

#   下ggplot画图代码为AI完成
ggplot(importance_df, aes(x = reorder(Variable, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Random Forest: Variable Importance (Gini Index)",
       x = "Variable", y = "Mean Decrease in Gini") +
  theme_minimal()

#   在测试集上进行预测
rf_test_prob <- predict(rf_model, newdata = test_data[, c(significant_vars)], type = "prob")[, 2]
rf_test_pred <- ifelse(rf_test_prob > 0.5, 1, 0)

#计算混淆矩阵
rf_confmat <- confusionMatrix(factor(rf_test_pred), factor(test_data$binary_quality))
#     打印混淆矩阵和准确率
print("RandomF Matrix:")
print(rf_confmat)

# 计算 ROC 曲线和 AUC
rf_roc <- roc(test_data$binary_quality, rf_test_prob)
rf_auc <- auc(rf_roc)
# 打印 AUC
print(paste("Random Forest AUC:", round(rf_auc, 4)))

# 绘制 ROC 曲线
#下ggplot画图代码为AI完成
ggroc(rf_roc) + 
  ggtitle("ROC Curve - Random Forest") +
  theme_minimal()

# 可视化其他评测指标（如准确率，F1-score等）
#下画图代码为AI完成
rf_perf_metrics <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity", "Precision", "F1-Score"),
  Value = c(rf_confmat$overall['Accuracy'],
            rf_confmat$byClass['Sensitivity'],
            rf_confmat$byClass['Specificity'],
            rf_confmat$byClass['Pos Pred Value'],
            rf_confmat$byClass['F1'])
)

# 绘制评测指标
#下ggplot画图代码为AI完成
ggplot(rf_perf_metrics, aes(x = Metric, y = Value, fill = Metric)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Random Forest: Performance Metrics", x = "Metric", y = "Value")

# 决策树建模
tree_model <- rpart(
  factor(binary_quality) ~ .,
  data = train_data[, c(significant_vars, "binary_quality")],
  method = "class",
  control = rpart.control(cp = 0.01)
)

#     可视化决策树
rpart.plot(tree_model, type = 2, extra = 106,
           fallen.leaves = TRUE, main = "Decision tree")
test_pred_class <- predict(tree_model, newdata = test_data[, significant_vars], type = "class")
test_pred_prob <- predict(tree_model, newdata = test_data[, significant_vars], type = "prob")[, 2]

conf_mat <- confusionMatrix(test_pred_class, factor(test_data$binary_quality))
print("混淆矩阵：")
print(conf_mat)

#ROC AUC
roc_obj <- roc(test_data$binary_quality, test_pred_prob)
auc_val <- auc(roc_obj)
print(paste("AUC:", round(auc_val, 4)))

#绘制 ROC 曲线
#下画图代码为AI完成
plot(roc_obj, col = "blue", lwd = 2, main = "ROC Curve - Decision Tree")
abline(a = 0, b = 1, lty = 2, col = "gray")
var_imp <- as.data.frame(tree_model$variable.importance)
var_imp$Variable <- rownames(var_imp)
colnames(var_imp) <- c("Importance", "Variable")

#下ggplot画图代码为AI完成
ggplot(var_imp, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "tomato") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Variable Importance - Decision Tree", x = "Variable", y = "Importance")

summary(tree_model)