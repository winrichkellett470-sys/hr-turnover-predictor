"""
train_model.py —— 训练冠军模型 (Stacking) 并保存到本地

使用方式：
    python train_model.py

运行完成后会在当前目录生成：
    - champion_model.pkl     (训练好的 Stacking 模型)
    - scaler.pkl             (特征标准化器)
    - feature_info.pkl       (特征名、部门列表等元信息)
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ==============================================================
# 1. 加载数据
# ==============================================================
candidate_paths = ["HR.csv", "./data/HR.csv", "../HR.csv"]
csv_path = next((p for p in candidate_paths if os.path.exists(p)), None)
if csv_path is None:
    raise FileNotFoundError("找不到 HR.csv，请把它放到脚本同目录")

print(f"[1/4] 加载数据: {csv_path}")
df = pd.read_csv(csv_path)
print(f"      数据规模: {df.shape}")

# ==============================================================
# 2. 预处理 —— 保持和 notebook 完全一致
# ==============================================================
print("[2/4] 数据预处理（独热编码 + 标准化）...")

df_final = pd.get_dummies(df, drop_first=True)
X = df_final.drop('left', axis=1)
y = df_final['left']
feature_names = X.columns.tolist()

# 记录原始部门列表和薪资水平，供前端下拉框使用
departments = sorted(df['position'].unique().tolist())
salary_levels = ['low', 'medium', 'high']   # 固定顺序

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# ==============================================================
# 3. 训练 Stacking 冠军模型
# ==============================================================
print("[3/4] 训练 Stacking 集成模型（约 1-2 分钟）...")

base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False,
                          eval_metric='logloss', random_state=42, n_jobs=1)),
    ('gbdt', GradientBoostingClassifier(n_estimators=100, random_state=42)),
]
model = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    n_jobs=1,
)
model.fit(X_train_s, y_train)

# 评估确认
y_pred = model.predict(X_test_s)
y_score = model.predict_proba(X_test_s)[:, 1]
print(f"      ACC: {accuracy_score(y_test, y_pred):.4f}")
print(f"      AUC: {roc_auc_score(y_test, y_score):.4f}")

# ==============================================================
# 4. 保存模型与元信息
# ==============================================================
print("[4/4] 保存模型文件...")

with open('champion_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# 参考基准（用于综合得分的归一化/行业基准对比）
reference_stats = {
    'satisfaction_mean': float(df['satisfaction_level'].mean()),
    'evaluation_mean': float(df['last_evaluation'].mean()),
    'project_mean': float(df['number_project'].mean()),
    'hours_mean': float(df['average_montly_hours'].mean()),
    'tenure_mean': float(df['time_spend_company'].mean()),
}

feature_info = {
    'feature_names': feature_names,       # 独热编码后的 18 列列名
    'departments': departments,           # 原始部门列表
    'salary_levels': salary_levels,
    'reference_stats': reference_stats,   # 行业基准
}
with open('feature_info.pkl', 'wb') as f:
    pickle.dump(feature_info, f)

print()
print("✅ 训练完成！生成的文件：")
print("   - champion_model.pkl")
print("   - scaler.pkl")
print("   - feature_info.pkl")
print()
print("下一步：运行 streamlit run hr_agent_app.py 启动智能体")
