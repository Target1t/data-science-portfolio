# ==============================================================================
# 1. 数据准备
# ==============================================================================
library(ggplot2)
library(dplyr)
library(scales)

data <- read.csv("Byssinosis.csv")

# 预处理
data$Total <- data$Byssinosis + data$Non.Byssinosis
data$Rate <- data$Byssinosis / data$Total

# 设置因子水平
data$Workspace <- factor(data$Workspace, levels = c(1, 2, 3), 
                         labels = c("1 (Most Dusty)", "2 (Less Dusty)", "3 (Least Dusty)"))
data$Employment <- factor(data$Employment, levels = c("<10", "10~19", ">=20"))
data$Smoking <- factor(data$Smoking, levels = c("No", "Yes"))
data$Sex <- factor(data$Sex)

# ==============================================================================
# 2. EDA: 1张频数图 + 3张比率图
# ==============================================================================

# --- Chart 1: [频数] 病例的绝对分布 (The Burden) ---
# 价值：直观展示绝大多数病人来自车间 1。尽管车间 3 人很多，但病人很少。
# 聚合病例数
agg_cases <- data %>%
  group_by(Workspace, Smoking) %>%
  summarise(Cases = sum(Byssinosis), .groups = 'drop')

p1 <- ggplot(agg_cases, aes(x = Workspace, y = Cases, fill = Smoking)) +
  geom_bar(stat = "identity", position = "stack", width = 0.6) +
  # 在堆叠柱状图中间添加具体的病例数
  geom_text(aes(label = Cases), position = position_stack(vjust = 0.5), color = "white", size = 4) +
  # 在柱子顶端添加每组的总数
  stat_summary(fun = sum, aes(label = ..y.., group = Workspace), geom = "text", vjust = -0.5, size = 5) +
  scale_fill_manual(values = c("No" = "#a6cee3", "Yes" = "#e31a1c")) +
  labs(title = "1. Disease Burden: Count of Cases by Workspace",
       subtitle = "Key Insight: Workspace 1 generates the vast majority (105) of all cases.",
       y = "Number of Byssinosis Cases", x = "Workspace") +
  theme_classic() +
  ylim(0, 120) # 稍微增加y轴空间给标签

# --- Chart 2: [比率] 风险等级 (The Risk) ---
# 价值：消除样本量影响，展示真实的危险程度。
agg_risk <- data %>%
  group_by(Workspace, Smoking) %>%
  summarise(Byssinosis = sum(Byssinosis), Total = sum(Total), .groups = 'drop') %>%
  mutate(Rate = Byssinosis / Total)

p2 <- ggplot(agg_risk, aes(x = Workspace, y = Rate, fill = Smoking)) +
  geom_bar(stat = "identity", position = position_dodge(), color = "black") +
  geom_text(aes(label = percent(Rate, accuracy = 0.1)), 
            position = position_dodge(width = 0.9), vjust = -0.5, size = 3.5) +
  scale_fill_manual(values = c("No" = "#a6cee3", "Yes" = "#e31a1c")) +
  labs(title = "2. Risk Level: Prevalence Rate",
       subtitle = "Even Non-Smokers in Workspace 1 (11.6%) are at higher risk than Smokers elsewhere.",
       y = "Prevalence Rate", x = "Workspace") +
  theme_minimal()

# --- Chart 3: [比率] 工龄趋势 (Duration) ---
# 价值：展示在脏环境下，时间如何放大风险。
agg_emp <- data %>%
  group_by(Employment, Workspace) %>%
  summarise(Byssinosis = sum(Byssinosis), Total = sum(Total), .groups = 'drop') %>%
  mutate(Rate = Byssinosis / Total)

p3 <- ggplot(agg_emp, aes(x = Employment, y = Rate, group = Workspace, color = Workspace)) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 3) +
  geom_text(aes(label = percent(Rate, accuracy = 0.1)), vjust = -1, size = 3, show.legend = FALSE) +
  scale_color_brewer(palette = "OrRd", direction = -1) +
  labs(title = "3. Duration Effect: Rate by Employment Length",
       subtitle = "Risk persists and varies over time, most severely in Workspace 1.",
       y = "Prevalence Rate", x = "Years of Employment") +
  theme_bw()

# --- Chart 4: [频数比例] 性别分布 (Confounding Check) ---
# 价值：揭示潜在的性别分配不均，解释为何男性看起来更易感。
agg_sex <- data %>%
  group_by(Workspace, Sex) %>%
  summarise(Total = sum(Total), .groups = 'drop') %>%
  group_by(Workspace) %>%
  mutate(Prop = Total / sum(Total))

p4 <- ggplot(agg_sex, aes(x = Workspace, y = Prop, fill = Sex)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = percent(Prop, accuracy = 1)), 
            position = position_stack(vjust = 0.5), color = "white") +
  scale_fill_manual(values = c("F" = "pink3", "M" = "steelblue")) +
  labs(title = "4. Confounding Check: Worker Distribution by Sex",
       subtitle = "Males are disproportionately assigned to the dangerous Workspace 1 (91%).",
       y = "Proportion of Workforce", x = "Workspace") +
  theme_light()

# 打印图表
print(p1)
print(p2)
print(p3)
print(p4)

# ==============================================================================
# 3. 统计建模 (BIC)
# ==============================================================================

# 模型部分保持不变，使用 BIC 选择最佳模型
y <- cbind(data$Byssinosis, data$Non.Byssinosis)
full_model <- glm(y ~ Workspace * (Employment + Smoking + Sex + Race), 
                  family = binomial(link = "logit"), data = data)
model_bic <- step(full_model, direction = "both", k = log(nrow(data)), trace = 0)

print("---------------- Final Model Summary ----------------")
summary(model_bic)
print("---------------- Odds Ratios ----------------")
# 第一步：先用 cbind 把系数(coef)和置信区间(confint)拼起来，
# 并在这里给系数列起个名字叫 "OR"
# 创建一个数据框或向量，并命名
data.frame(OR = exp(coef(model_bic)))
