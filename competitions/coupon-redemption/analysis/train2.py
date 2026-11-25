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
train = pd.read_csv('train2.csv')

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
    X, y, test_size=0.1, random_state=42
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42  # 最终比例: 80%训练, 10%验证, 10%测试
)

# 定义采样策略
over = SMOTE(sampling_strategy={1: int(len(y_train)*0.3)}, k_neighbors=5)
under = RandomUnderSampler(sampling_strategy='majority')

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
    'objective': 'binary:logistic',
    'max_depth': 6,  # 稍加深以提高模型容量
    'scale_pos_weight': 2,  # 根据类1的样本比例调整，假设类1是少数类
    'learning_rate': 0.05,  # 更小的学习率
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'gpu_id': 0,
    'max_bin': 256,  # 减少GPU内存使用
    'single_precision_histogram': True,  # 使用单精度浮点数
    'eval_metric': ['error', 'logloss', 'auc']  # 添加AUC指标
}

# 5. 增量训练（节省内存）
evals_result = {}  # 添加这行来收集评估结果
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,  # 设置较大值，靠早停控制
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=20,  # 20轮无提升则停止
    verbose_eval=10,  # 每10轮输出一次日志
    evals_result=evals_result  # 添加这个参数
)

# 可视化学习过程
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(evals_result['train']['error'], label='Train Error')
plt.plot(evals_result['val']['error'], label='Validation Error')
plt.xlabel('Boosting Rounds')
plt.ylabel('Error Rate')
plt.title('Training vs Validation Error')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(evals_result['train']['logloss'], label='Train Logloss')
plt.plot(evals_result['val']['logloss'], label='Validation Logloss')
plt.xlabel('Boosting Rounds')
plt.ylabel('Log Loss')
plt.title('Training vs Validation Log Loss')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# 特征重要性分析
importance = model.get_score(importance_type='weight')
importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 8))
xgb.plot_importance(model, max_num_features=15, importance_type='weight')
plt.title('Feature Importance (Weight)')
plt.show()

print("\nTop重要特征:")
for feat, score in importance[:9]:
    print(f"{feat}: {score:.2f}")

# 过拟合检查代码
train_pred = (model.predict(dtrain) > 0.5).astype(int)
val_pred = (model.predict(dval) > 0.5).astype(int)

print("\n训练集性能:")
print(classification_report(y_train, train_pred, target_names=['未使用', '已使用']))
print("验证集性能:")
print(classification_report(y_val, val_pred, target_names=['未使用', '已使用']))

# 计算训练集和验证集的AUC差异
from sklearn.metrics import roc_auc_score
train_auc = roc_auc_score(y_train, model.predict(dtrain))
val_auc = roc_auc_score(y_val, model.predict(dval))
print(f"\n训练集AUC: {train_auc:.4f}, 验证集AUC: {val_auc:.4f}, 差异: {train_auc-val_auc:.4f}")

# 判断过拟合标准
if train_auc - val_auc > 0.1:  # AUC差异大于0.1
    print("\n警告：模型可能过拟合（训练集性能明显优于验证集）")
elif train_auc - val_auc > 0.05:
    print("\n注意：模型有轻微过拟合迹象")
else:
    print("\n模型没有明显过拟合迹象")

# 6. 模型评估
y_pred = (model.predict(dtest) > 0.5).astype(int)  # 添加阈值处理

print("分类报告：")
print(classification_report(y_test, y_pred, target_names=['未使用', '已使用']))

print("\n已核销的混淆矩阵：")
print(confusion_matrix(y_test, y_pred, labels=[1, 0]))  # 直接使用原始标签

