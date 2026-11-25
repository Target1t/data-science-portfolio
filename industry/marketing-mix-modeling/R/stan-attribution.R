#### Step 9: 贝叶斯模型借助Stan语言编写

## 确定是否限制baseline最大比例
'red'
is_limited <- FALSE   # 选择FALSE的话，默认不限制baseline的占比，如果选择TRUE，则需要对baseline进行占比的限制

## 确保dt_aggr和dt_selected_bayesian顺序相同
dt_aggr1 <- read.csv(paste0(robyn_output_folder,'pareto_aggregated.csv'))
dt_aggr1 <- dt_aggr1[dt_aggr1$solID == solID_val, c("rn", "coef")]
dt_aggr <- dt_aggr1
for(i in 2:nrow(dt_aggr)){
  dt_aggr[i,] = dt_aggr1[dt_aggr1$rn == colnames(dt_selected_bayesian)[i+1],]
}

## 提取rn列名（假设rn列是第一列）
rn_columns <- dt_aggr$rn
rn_columns <- rn_columns[!grepl("Intercept", rn_columns)]  # 去掉特殊列
num_media <- length(rn_columns)

## 设置baseline组合中的所有列名
'red'
baseline_names <- c('(Intercept)', 'trend_b', 'season_b', 'holiday_b')   # 根据实际填写，如果未启用season或holiday则需要将'season_b'/'holiday_b'删除
num_baseline <- length(baseline_names)
baseline <- dt_aggr[dt_aggr$rn %in% baseline_names,]
others <- dt_aggr[!dt_aggr$rn %in% baseline_names,]


## 设置baseline的最大比例
'red'
K <- 0.7   # 如果限制baseline比例，需要在这里填写希望得到的baseline的范围，如果是70%，则填0.7

## 根据是否限制baseline最大比例得到stan代码
if(!is_limited){
  ## 构建data部分
  data_part <- c(
    "data {",
    "    int N; // 数据点的总数",
    "    int num_media; // 媒体变量的数量,按数据中的媒体名称+_b/_c进行调整",
    paste0("    vector[N] ", rn_columns, ';', collapse = "\n"),
    "    vector[N] Y;//Response variable - KPI, 长度为N",
    "}"
  )
  
  ## 构建parameters部分
  parameters_part <- c(
    "parameters {",
    "    real a_field; //intercept先验均值",
    "    real<lower=0> a; //intercept, 约束为非负",
    paste0("    real<", sapply(1:num_media, function(x){
      switch(substr(dt_aggr$rn[-1][x],nchar(dt_aggr$rn[-1][x])-1,nchar(dt_aggr$rn[-1][x])), 
             '_B' = ifelse(dt_aggr$coef[-1][x]>=0,"lower","upper"), 
             '_b' = ifelse(dt_aggr$coef[-1][x]>=0,"lower","upper"), 
             '_S' = "lower", 
             '_s' = "lower",
             '_I' = "lower",
             '_i' = "lower",
             '_C' = "upper",
             '_c' = "upper")
    }), "=0> b_", 1:num_media, "; //", rn_columns),
    "\n    vector[num_media] b_field; //每个媒体beta的先验均值",
    "    real<lower=0> s_y;  //Y的先验标准差",
    "    vector<lower=0>[num_media] s_b;//每个媒体beta的先验标准差",
    "    real<lower=0> s_a; //intercept的先验标准差",
    "}"
  )
  
  ## 构建transformed parameters部分
  transformed_part <- paste0(
    "transformed parameters {\n    vector[N] result = a",
    paste0("+b_", 1:num_media, "*", rn_columns, "[1:N]", collapse = ""),
    ";", "\n}", collapse = ""
  )
  
  ## 构建model部分
  model_part <- c(
    "model {",
    "    a ~ normal(a_field, s_a); //intercept的先验分布",
    paste0("    b_", 1:num_media, " ~ normal(b_field[", 1:num_media, "], s_b[", 1:num_media, "]);"),
    "    Y ~ normal(result, s_y);",
    "}"
  )
  
  ## 合并所有部分
  stan_model <- paste(c(data_part, parameters_part, transformed_part, model_part), collapse = '\n')
  ## 输出结果
  cat(stan_model)
}else{
  ## 构建data部分
  data_part <- c(
    "data {",
    "    int N; // 数据点的总数",
    "    int num_media; // 媒体变量的数量,按数据中的媒体名称+_b/_c进行调整",
    "    real<lower=0> K;  // 自定义上限K",
    paste0("    vector[N] ", rn_columns, ';', collapse = "\n"),
    "    vector[N] Y;//Response variable - KPI, 长度为N",
    "}"
  )
  
  ## 构建parameters部分
  parameters_part <- c(
    "parameters {",
    paste0("    simplex[", num_baseline, "] raw_weights;  // 权重向量（和为1）"),
    "    real<lower=0> scale;      // 缩放因子",
    paste0("    real<", sapply(1:(num_media-num_baseline+1), function(x){
      switch(substr(others$rn[x],nchar(others$rn[x])-1,nchar(others$rn[x])), 
             '_B' = ifelse(others$coef[x]>=0,"lower","upper"), 
             '_b' = ifelse(others$coef[x]>=0,"lower","upper"), 
             '_S' = "lower", 
             '_s' = "lower",
             '_C' = "upper",
             '_c' = "upper")
    }), "=0> b_", 1:(num_media-num_baseline+1), "; //", as.character(others$rn)),
    "\n    vector[num_media] b_field; //每个媒体beta的先验均值",
    "    real<lower=0> s_y;  //Y的先验标准差",
    "    vector<lower=0>[num_media] s_b;//每个媒体beta的先验标准差",
    "    real<lower=0> s_a; //intercept的先验标准差",
    "}"
  )
  
  ## 构建transformed parameters部分
  transformed_part <- paste0(
    "transformed parameters {", "\n", "    // 按K缩放权重得到四参数\n",
    paste0("    real ", ifelse(baseline$rn == "(Intercept)", "a", "b_"), c("",(num_media-num_baseline+2):num_media), " = raw_weights[", 1:num_baseline, "] * K * sum(Y) / ", ifelse(baseline$rn == "(Intercept)", "N", paste0("sum(", baseline$rn, ")")), ";", collapse = "\n"),
    "\n", 
    "    vector[N] result = a",
    paste0("+b_", 1:num_media, "*", rn_columns, "[1:N]", collapse = ""),
    ";", "\n}", collapse = ""
  )
  
  ## 构建model部分
  model_part <- c(
    "model {",
    paste0("    raw_weights ~ dirichlet(rep_vector(1, ", num_baseline, "));  // 权重先验"), 
    "    scale ~ exponential(0.1);                   // 缩放因子先验",
    paste0("    b_", 1:(num_media-num_baseline+1), " ~ normal(b_field[", 1:(num_media-num_baseline+1), "], s_b[", 1:(num_media-num_baseline+1), "]);"),
    "    Y ~ normal(result, s_y);",
    "}"
  )
  
  ## 合并所有部分
  stan_model <- paste(c(data_part, parameters_part, transformed_part, model_part), collapse = '\n')
  ## 输出结果
  cat(stan_model)
}
