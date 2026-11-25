import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 加载数据

train = pd.read_csv('train.csv')

# 1. 准备数据
feature_cols = [col for col in train.columns if col not in ['Coupon_status']]
X = train[feature_cols]
y = train['Coupon_status']

# 2. 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X = pd.DataFrame(X_scaled, columns=feature_cols)

# 3. 划分训练验证测试集 (添加验证集用于早停)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.05, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.05, random_state=42  # 最终比例: 90%训练, 5%验证, 5%测试
)

# 定义采样策略
over = SMOTE(sampling_strategy={1: 570000*2},  # 将类1样本翻倍
             k_neighbors=5, random_state=42)
under = RandomUnderSampler(sampling_strategy={0: 570000,  # 类0采样到与类1相同
                                             2: 570000*2},  # 类2采样到类1的2倍
                           random_state=42)

# 创建采样管道
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)

# 应用采样
X_train, y_train = pipeline.fit_resample(X_train, y_train)

# 转换为DMatrix格式（更高效）
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# 4. 构建优化后的XGBoost参数
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'scale_pos_weight': [1, 10, 0.5],  # 给类1(实际1)3倍权重，类2(实际2)减半权重
    'max_depth': 6,  # 稍加深以提高模型容量
    'learning_rate': 0.05,  # 更小的学习率
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0,
    'max_bin': 256,  # 减少GPU内存使用
    'single_precision_histogram': True,  # 使用单精度浮点数
    'eval_metric': ['merror', 'mlogloss', 'auc']  # 添加AUC指标
}

# 5. 增量训练（节省内存）
evals_result = {}  # 先创建空字典存储结果
model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,  # 设置较大值，靠早停控制
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=20,  # 20轮无提升则停止
    verbose_eval=10,  # 每10轮输出一次日志
    evals_result=evals_result  # 通过参数收集评估结果
)

# 新增：获取训练过程指标
epochs = len(evals_result['train']['merror'])
x_axis = range(0, epochs)

# 绘制对数损失变化曲线
plt.subplot(1, 2, 2)
plt.plot(x_axis, evals_result['train']['mlogloss'], label='Train')
plt.plot(x_axis, evals_result['val']['mlogloss'], label='Validation')
plt.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.grid()
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x_axis, evals_result['train']['merror'], label='Train')
plt.plot(x_axis, evals_result['val']['merror'], label='Validation')

plt.tight_layout()
plt.show()

# 6. 模型评估
y_pred = model.predict(dtest)

print("分类报告：")
print(classification_report(y_test, y_pred, target_names=['未使用', '已使用', '其他']))

print("\n类别2的混淆矩阵：")
print(confusion_matrix(y_test == 2, y_pred == 2, labels=[True, False]))
