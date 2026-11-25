import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
# ----------------------------
# 数据加载与基础预处理（不涉及目标变量）
# ----------------------------
def load_data():
    df_female = pd.read_excel(r"D:\FDSM\2025Fall@UCD\建模大赛2025\C题\附件.xlsx", sheet_name='女胎检测数据')
    return df_female

def basic_preprocess(df):
    """仅处理与目标变量无关的预处理"""
    # 删除末次月经为空
    df = df.dropna(subset=['末次月经'])
    
    # 孕周格式转换（不依赖其他特征）
    def transform_yunzhou(data):
        i = 0
        weeks = 0
        while data[i]!='w' and data[i]!='W':
            i+=1
        weeks += int(data[:i])
        if data[-1]=='w' or data[-1]=='W':
            return weeks
        if data[i+1]=='+':
            weeks += int(data[i+2])/7
        return weeks
    
    df['检测孕周'] = df['检测孕周'].apply(transform_yunzhou)
    return df

# ----------------------------
# 数据划分（先划分再分别处理）
# ----------------------------
df_female = load_data()
df_female = basic_preprocess(df_female)

# 定义特征和目标
features = [
    '21号染色体的Z值', '18号染色体的Z值', '13号染色体的Z值',
    'X染色体的Z值', 'X染色体浓度',
    '21号染色体的GC含量', '18号染色体的GC含量', '13号染色体的GC含量',
    '被过滤掉读段数的比例', '检测孕周', '孕妇BMI',
    '原始读段数','在参考基因组上比对的比例', '重复读段的比例','唯一比对的读段数'
]

# 先划分再分别处理
X = df_female[features]
y = df_female['染色体的非整倍体'].notnull().astype(int)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ----------------------------
# 训练集专用预处理
# ----------------------------
def train_preprocess(X, y):
    """仅在训练集上计算并应用预处理"""
    X = X.copy()
    
    # 1. 3sigma原则处理
    for chrom in ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']:
        mu, std = X[chrom].mean(), X[chrom].std()
        X = X[(X[chrom] > mu-3*std) & (X[chrom] < mu+3*std)]
        y = y.loc[X.index]
    
    # 2. BMI的IQR处理（仅用训练集计算）
    BMI_Q1 = X['孕妇BMI'].quantile(0.25)
    BMI_Q3 = X['孕妇BMI'].quantile(0.75)
    BMI_IQR = BMI_Q3 - BMI_Q1
    X = X[
        (X['孕妇BMI'] > BMI_Q1-1.5*BMI_IQR) & 
        (X['孕妇BMI'] < BMI_Q3+1.5*BMI_IQR)
    ]
    y = y.loc[X.index]
    
    # 3. 缺失值填充（用训练集统计量）
    medians = X.median()
    X = X.fillna(medians)
    
    return X, y

# 应用训练集预处理
X_train_processed, y_train_processed = train_preprocess(X_train, y_train)

# ----------------------------
# 测试集预处理（使用训练集计算的参数）
# ----------------------------
def test_preprocess(X, train_medians, train_stats):
    """使用训练集计算的参数处理测试集"""
    X = X.copy()
    
    # 1. 3sigma过滤（使用训练集的mu和std）
    for chrom in ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']:
        mu, std = train_stats[chrom]['mean'], train_stats[chrom]['std']
        X = X[(X[chrom] > mu-3*std) & (X[chrom] < mu+3*std)]
    
    # 2. BMI过滤（使用训练集的IQR）
    BMI_Q1, BMI_Q3 = train_stats['BMI']['Q1'], train_stats['BMI']['Q3']
    BMI_IQR = BMI_Q3 - BMI_Q1
    X = X[
        (X['孕妇BMI'] > BMI_Q1-1.5*BMI_IQR) & 
        (X['孕妇BMI'] < BMI_Q3+1.5*BMI_IQR)
    ]
    
    # 3. 缺失值填充（使用训练集中位数）
    X = X.fillna(train_medians)
    
    return X

# 计算训练集统计量
train_stats = {
    chrom: {'mean': X_train[chrom].mean(), 'std': X_train[chrom].std()} 
    for chrom in ['13号染色体的Z值', '18号染色体的Z值', '21号染色体的Z值', 'X染色体的Z值']
}
train_stats['BMI'] = {
    'Q1': X_train['孕妇BMI'].quantile(0.25),
    'Q3': X_train['孕妇BMI'].quantile(0.75)
}
train_medians = X_train.median()

# 处理测试集
X_test_processed = test_preprocess(X_test, train_medians, train_stats)
y_test_processed = y_test.loc[X_test_processed.index]

# ----------------------------
# 模型训练与评估
# ----------------------------
rf = RandomForestClassifier(
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)

param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [8, 12, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    rf, param_grid, 
    cv=5, scoring='precision', n_jobs=-1
)
grid_search.fit(X_train_processed, y_train_processed)

best_rf = grid_search.best_estimator_
y_proba = best_rf.predict_proba(X_test_processed)[:, 1]

# 阈值优化
precision, recall, thresholds = precision_recall_curve(y_test_processed, y_proba)
optimal_idx = np.argmax(precision[:-1] + recall[:-1])  # 平衡precision和recall
optimal_threshold = thresholds[optimal_idx]
y_pred = (y_proba >= optimal_threshold).astype(int)

print("测试集样本量:", len(X_test_processed))
print("\n最佳参数:", grid_search.best_params_)
print(classification_report(y_test_processed, y_pred))

# ----------------------------
# 特征重要性可视化
# ----------------------------
plt.figure(figsize=(10,6))
pd.Series(best_rf.feature_importances_, index=X_train_processed.columns
         ).sort_values().plot.barh()
plt.title('特征重要性')
plt.tight_layout()
plt.show()