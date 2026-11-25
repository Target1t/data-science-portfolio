plot <- read.csv("D:\\FDSM\\2025Spring\\抽样调查\\小组作业\\shorebirds\\shorebirds.csv")
# 将 -9 视为缺失值
plot$intense[plot$intense == -9] <- NA
# 第一阶段所有 rapid 值
x_all <- plot$rapid
x_total <- sum(x_all)
# 第二阶段子样本（intense 不为 NA 的）
subsample <- subset(plot, !is.na(intense))
x_sub <- subsample$rapid
y_sub <- subsample$intense
# 计算比率估计器 R_hat
R_hat <- sum(y_sub) / sum(x_sub)
# 使用比率估计器估计总巢数
Y_hat <- R_hat * x_total
# 输出结果
cat("Estimated total number of nests (two-phase ratio estimator):", round(Y_hat), "\n")