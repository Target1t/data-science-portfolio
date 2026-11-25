df = read.csv("D:\\FDSM\\2025Fall@UCD\\建模大赛2025\\C题\\md.csv")
df
# 安装需要的包（只需运行一次）
# install.packages("lme4")
# install.packages("MuMIn")

library(lme4)
library(MuMIn)

# 假设你的 dataframe 名字是 df
# 列名: 孕妇代码, 检测孕周, 孕妇BMI, Y染色体浓度

# 构建混合效应模型
# 固定效应：检测孕周 + 孕妇BMI
# 随机效应：孕妇代码（截距随机）
model <- lmer(`Y染色体浓度` ~ `检测孕周` + `孕妇BMI` + (1 | `孕妇代码`), data = df)

# 模型结果
summary(model)

# 计算 R²
r2 <- r.squaredGLMM(model)
print(r2)

op <- par(mfrow = c(2,2))

# 2.1 残差与拟合值（看同方差性/模式）
plot(fitted(model), resid(model),
     xlab = "Fitted values", ylab = "Raw residuals")
abline(h = 0, lty = 2, col = "red")

# 2.2 标准化残差（异常值更敏感）
plot(fitted(model), resid(model, type = "pearson"),
     xlab = "Fitted values", ylab = "Pearson residuals")
abline(h = 0, lty = 2, col = "red")

# 2.3 QQ 图（残差正态性）
qqnorm(resid(model)); qqline(resid(model), col = "red")

# 2.4 随机效应（BLUP）正态性
re <- ranef(model, condVar = TRUE)$`孕妇代码`[,1]
qqnorm(re, main = "Random intercepts (孕妇代码)"); qqline(re, col = "blue")

par(op)
m_full <- lmer(`Y染色体浓度` ~ `检测孕周` + `孕妇BMI` + (1 | `孕妇代码`), data = df)
m_null <- lmer(`Y染色体浓度` ~ 1 + (1 | `孕妇代码`), data = df)

anova(m_null, m_full)   # LRT，显著说明固定效应整体有用


