#!/usr/bin/env python
# coding: utf-8

# Step 1: 导入所需的库
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Step 2: 加载和准备数据
# 加载数据
with open('.json') as f:
    data_a = json.load(f)

with open('.json') as f:
    data_c = json.load(f)

# 转换成pandas DataFrame
df_a = pd.DataFrame(data_a).transpose()
df_c = pd.DataFrame(data_c, index=["category"]).transpose()

# 标签转换：从1和2转换为0和1
# df_c['category'] = df_c['category'] - 1

# 标签转换:3定义为1,而0 1 2定义为0
df_c['category'] = df_c['category'].replace({3: 1, 0: 0, 1: 0, 2: 0})

# 合并DataFrame
df = df_a.join(df_c, how='inner')


# 提取特征和标签
X = df[['mean']].values
y = df['category'].values

# Step 3:划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=35)

# Step 4: 特征预处理
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: 处理类别不平衡
ros = RandomOverSampler(random_state=42)
X_train_ros, y_train_ros = ros.fit_resample(X_train_scaled, y_train)

# Step 6: 构建并训练BP神经网络
# 创建更复杂的神经网络模型
np.random.seed(6)
model = Sequential()
model.add(Dense(256, input_dim=1, activation='relu'))  # 增加神经元数量
model.add(Dense(128, activation='relu'))  # 添加更多层
model.add(Dense(64, activation='relu'))  # 添加更多层
model.add(Dense(1, activation='sigmoid'))

# 设置优化器
optimizer = Adam(lr=0.0001)

# 编译模型
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 训练模型，移除早停法，增加训练周期，这里我们不使用validation_split以专注于训练集
history = model.fit(X_train_ros, y_train_ros, epochs=800, batch_size=33, verbose=0)

# Step 7: 评估模型和计算评价指标
# 评估模型
train_loss, train_accuracy = model.evaluate(X_train_ros, y_train_ros, verbose=0)
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)

print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")

# 预测概率
y_train_pred = model.predict(X_train_ros).ravel()
y_test_pred = model.predict(X_test_scaled).ravel()

# 计算ROC曲线
fpr_train, tpr_train, thresholds_train = roc_curve(y_train_ros, y_train_pred)
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_pred)
auc_train = auc(fpr_train, tpr_train)
auc_test = auc(fpr_test, tpr_test)

# Step 8: 绘制ROC曲线
plt.figure()
plt.plot(fpr_train, tpr_train, 'b-', label = 'Train AUC: %.3f' % auc_train)
plt.plot(fpr_test, tpr_test, 'g-', label = 'Test AUC: %.3f' % auc_test)
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc = 'lower right')
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)

plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.legend(loc="lower right")
plt.show()

