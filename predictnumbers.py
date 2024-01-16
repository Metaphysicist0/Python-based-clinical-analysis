# 导入需要的库
import json
import numpy as np
import math
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # 导入评价指标
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

# 读取json数据
with open('results.json') as f:
    result = json.load(f)

# 这里改因素
with open('snumber.json') as f:
    s_num = json.load(f)

with open('stage.json') as f:
    stages = json.load(f)

# 提取特征
x = []
y = []
s = []
u = []
filtered_results = {}

for id, data in result.items():
    stage = stages.get(id)
    if stage == 'D' or stage == 'E' or stage == 'C' or stage == 'B' or stage == 'A':
        filtered_results[id] = data


for identifier, value in filtered_results.items():
    mean = value['mean']
    num = s_num.get(identifier, 0)
    x.append(mean)
    y.append(num)

# 转换为numpy数组
x = np.array(x).reshape(-1, 1)
y = np.array(y)

# 重新划分训练集测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# 创建人工神经网络模型，使用参数
model = MLPRegressor(hidden_layer_sizes=(500, 500), activation='relu', solver='adam', alpha=0.0001, max_iter=500)

# 训练模型
model.fit(x_train, y_train)

# 预测测试集
y_pred = model.predict(x_test)
# 四舍五入取整
y_pred = np.round(y_pred)

# 计算评价指标
mse = mean_squared_error(y_test, y_pred)  # 均方误差
mae = mean_absolute_error(y_test, y_pred)  # 平均绝对误差

# 计算预测准确率
y_mean = np.mean(y)  # 真实结果的平均值
baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_mean))  # 基准误差

# 计算平均绝对百分比误差
mape = np.mean(np.abs((y_test - y_pred) / y_test))

# 计算预测准确率
accuracy = 1 - (mape / baseline_mae)


# 打印预测结果和真实结果
print('预测结果:', y_pred)
print('真实结果:', y_test)

# 打印评价指标
print('均方误差:', mse)
print('平均绝对误差:', mae)

# 打印预测准确率
print('预测准确率:', accuracy)

# 打印网络参数
print('网络参数:')
print(model.get_params())  # 获取模型的所有参数

# 打印训练集和测试集的大小
print('训练集大小:', len(x_train))
print('测试集大小:', len(x_test))

# 打印神经元个数
print('神经元个数:')
print(model.n_outputs_)  # 输出层神经元个数
print(model.n_layers_)  # 网络层数（包括输入层和输出层）
print(model.hidden_layer_sizes)   # 隐藏层神经元个数

'''

# 导入需要的库
import json
import numpy as np
import math
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error  # 导入评价指标
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
# 导入sklearn.preprocessing模块
from sklearn.preprocessing import MinMaxScaler

# 读取json数据
with open('results.json') as f:
    result = json.load(f)


# 这里改因素
with open('snumber.json') as f:
    s_num = json.load(f)

with open('stage.json') as f:
    stages = json.load(f)

# 提取特征
x = []
y = []
s = []
u = []
filtered_results = {}

for id, data in result.items():
    stage = stages.get(id)
    if stage == 'D':
        filtered_results[id] = data


for identifier, value in filtered_results.items():
    mean = value['mean']
    num = s_num.get(identifier, 0)
    x.append(mean)
    y.append(num)

# 转换为numpy数组
x = np.array(x).reshape(-1, 1)
y = np.array(y)

# 创建一个MinMaxScaler对象，并用它来拟合x和y的数据
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_x.fit(x)
scaler_y.fit(y.reshape(-1, 1))

# 使用MinMaxScaler对象的transform方法，将x和y的数据转换为归一化后的数据
x_scaled = scaler_x.transform(x)
y_scaled = scaler_y.transform(y.reshape(-1, 1)).ravel()

# 重新划分训练集测试集
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, random_state=42)
# 创建人工神经网络模型，使用参数
model = MLPRegressor(hidden_layer_sizes=(500, 500), activation='tanh', solver='adam', alpha=0.01, max_iter=1000)

# 训练模型
model.fit(x_train, y_train)

# 预测测试集
y_pred = model.predict(x_test)

# 计算评价指标
mse = mean_squared_error(y_test, y_pred)  # 均方误差
mae = mean_absolute_error(y_test, y_pred)  # 平均绝对误差

# 计算预测准确率
y_mean = np.mean(y_scaled)  # 真实结果的平均值
baseline_mae = mean_absolute_error(y_test, np.full_like(y_test, y_mean))  # 基准误差

# 计算平均绝对百分比误差
mape = np.mean(np.abs((y_test - y_pred) / y_test))

# 计算预测准确率
accuracy = 1 - (mape / baseline_mae)


# 打印预测结果和真实结果
print('预测结果:', y_pred)
print('真实结果:', y_test)

# 打印评价指标
print('均方误差:', mse)
print('平均绝对误差:', mae)

# 打印预测准确率
print('预测准确率:', accuracy)

# 打印网络参数
print('网络参数:')
print(model.get_params())  # 获取模型的所有参数

# 打印训练集和测试集的大小
print('训练集大小:', len(x_train))
print('测试集大小:', len(x_test))

# 打印神经元个数
print('神经元个数:')
print(model.n_outputs_)  # 输出层神经元个数
print(model.n_layers_)  # 网络层数（包括输入层和输出层）
print(model.hidden_layer_sizes)   # 隐藏层神经元个数

from sklearn.metrics import roc_curve, auc

# 计算特异性和灵敏度
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# 计算AUC值
roc_auc = auc(fpr, tpr)

# 绘制ROC曲线
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
'''
