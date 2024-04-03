import pandas as pd

import torch.nn as nn

import torch

import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# 读取数据
data = pd.read_excel(r"G:\OneDrive\Desktop\Survival.xlsx")
# 数据预处理
X = data.iloc[:, 1:4].values  # 特征：死亡与否、病理类型、存活时间
y = data.iloc[:, -1].values    # 标签：存活时间

# 归一化存活时间
min_time = y.min()
max_time = y.max()
normalized_time = (y - min_time) / (max_time - min_time)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 将数据转换为PyTorch张量并增加一个维度
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).unsqueeze(1)  # 在第二个维度上增加一个维度
y_tensor = torch.tensor(normalized_time, dtype=torch.float32).view(-1, 1)

# 划分训练集和测试集
X_train_tensor, X_test_tensor, y_train_tensor, y_test_tensor = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 训练模型
def train_model(model, X_train, y_train, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

input_size = X_train_tensor.shape[2]
hidden_size = 1000
num_layers = 4
model = LSTMModel(input_size, hidden_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_model(model, X_train_tensor, y_train_tensor, criterion, optimizer)


# 测试模型
def test_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predicted = model(X_test).numpy()
    # 反归一化预测结果
    predicted_time = predicted * (max_time - min_time) + min_time

    # 打印预测结果和真实值
    print("Predicted Survival Time:")
    print(predicted_time)
    print("True Survival Time:")
    print(y_test.numpy() * (max_time - min_time) + min_time)

    return predicted_time


predicted_test = test_model(model, X_test_tensor, y_test_tensor)

# 评估模型
mse_test = mean_squared_error(y_test_tensor.numpy(), predicted_test)
r2_test = r2_score(y_test_tensor.numpy(), predicted_test)
print(f'Test Mean Squared Error: {mse_test:.4f}')
print(f'Test R^2 Score: {r2_test:.4f}')
# 绘制测试集的预测结果
plt.figure(figsize=(10, 6))
plt.plot(y_test_tensor.numpy(), label='True')
plt.plot(predicted_test, label='Predicted')
plt.legend()
plt.title('Survival Time Prediction - Test Set')
plt.xlabel('Sample')
plt.ylabel('Survival Time')
plt.show()

# 绘制各个类别的生存预测指标图
unique_categories = np.unique(X[:, 1])
plt.figure(figsize=(10, 6))
# 定义颜色列表

colors = ['blue', 'green', 'red', 'purple']
for i, category in enumerate(unique_categories):
    category_indices = np.where(X[:, 1] == category)[0]
    X_category = X_scaled[category_indices]
    y_category = y[category_indices]

    X_category_tensor = torch.tensor(X_category, dtype=torch.float32).unsqueeze(1)
    y_category_tensor = torch.tensor(y_category, dtype=torch.float32).view(-1, 1)

    predicted_category = test_model(model, X_category_tensor, y_category_tensor)
    sorted_indices = np.argsort(predicted_category[:, 0])
    survival_prob = np.linspace(1, 0, len(predicted_category), endpoint=False)

    # 仅绘制前2000天的数据
    plt.plot(predicted_category[sorted_indices][:2000], survival_prob[:2000], color=colors[i], label=f'Category {int(category)} - Survival Probability')

plt.xlabel('Predicted Survival Time (First 2000 Days)')
plt.ylabel('Survival Probability')
plt.title('Predicted Survival Probability Curve (First 2000 Days)')
plt.legend()
plt.tight_layout()
plt.show()
