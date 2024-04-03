import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


actual_survival = np.array([])

predicted_survival = np.array([])

# 计算均值
actual_mean = np.mean(actual_survival)
predicted_mean = np.mean(predicted_survival)

# 计算标准差
actual_std = np.std(actual_survival)
predicted_std = np.std(predicted_survival)

# 计算中位数
actual_median = np.median(actual_survival)
predicted_median = np.median(predicted_survival)

# 计算四分位数
actual_q1, actual_q3 = np.percentile(actual_survival, [25, 75])
predicted_q1, predicted_q3 = np.percentile(predicted_survival, [25, 75])

# 计算95%置信区间
actual_ci = stats.norm.interval(0.95, loc=actual_mean, scale=actual_std/np.sqrt(len(actual_survival)))
predicted_ci = stats.norm.interval(0.95, loc=predicted_mean, scale=predicted_std/np.sqrt(len(predicted_survival)))

# 计算皮尔逊相关系数
pearson_corr, _ = stats.pearsonr(actual_survival, predicted_survival)

# 计算斯皮尔曼等级相关系数
spearman_corr, _ = stats.spearmanr(actual_survival, predicted_survival)

# 计算肯德尔秩相关系数
kendall_corr, _ = stats.kendalltau(actual_survival, predicted_survival)

# 计算R²、MAE和RMSE
r2 = r2_score(actual_survival, predicted_survival)
mae = mean_absolute_error(actual_survival, predicted_survival)
rmse = np.sqrt(mean_squared_error(actual_survival, predicted_survival))

# Bland-Altman分析
def bland_altman_plot(data1, data2, *args, **kwargs):
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    md = np.mean(diff)
    sd = np.std(diff, axis=0)

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md, color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    plt.xlabel('Mean of Actual and Predicted Survival Time', fontsize=20)
    plt.ylabel('Difference between Actual and Predicted Survival Time', fontsize=20)
    plt.title('Bland-Altman Plot', fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

# 设置Nature Medicine风格的绘图参数
sns.set_style("ticks")
sns.set_context("poster", font_scale=1.5)
plt.figure(figsize=(16, 12))
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5

# 绘制散点图和回归线
sns.regplot(x=actual_survival, y=predicted_survival, color='#2C73B4', scatter_kws={"s": 80, "alpha": 0.8}, line_kws={"color": "#F05023", "linewidth": 4})

# 添加统计指标文本
plt.text(0.05, 0.95, f"Pearson Correlation: {pearson_corr:.4f}", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', color='#2C73B4')
plt.text(0.05, 0.90, f"Spearman Correlation: {spearman_corr:.4f}", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', color='#2C73B4')
plt.text(0.05, 0.85, f"Kendall Correlation: {kendall_corr:.4f}", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', color='#2C73B4')
plt.text(0.05, 0.80, f"R²: {r2:.4f}", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', color='#2C73B4')
plt.text(0.05, 0.75, f"MAE: {mae:.4f}", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', color='#2C73B4')
plt.text(0.05, 0.70, f"RMSE: {rmse:.4f}", transform=plt.gca().transAxes, fontsize=20, verticalalignment='top', color='#2C73B4')

# 设置标题和轴标签
plt.title("Actual vs. Predicted Survival Time", fontsize=28, fontweight='bold', color='#2C73B4')
plt.xlabel("Actual Survival Time (Days)", fontsize=24, color='#2C73B4')
plt.ylabel("Predicted Survival Time (Days)", fontsize=24, color='#2C73B4')

# 设置刻度标签的字体大小
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)

# 调整图形布局和显示
plt.tight_layout()
plt.show()

# 绘制Bland-Altman图
bland_altman_plot(actual_survival, predicted_survival, color='#F05023', alpha=0.8)

# 打印计算结果
print(f"Mean of Actual Survival Time: {actual_mean:.4f}")
print(f"Mean of Predicted Survival Time: {predicted_mean:.4f}")
print(f"Standard Deviation of Actual Survival Time: {actual_std:.4f}")
print(f"Standard Deviation of Predicted Survival Time: {predicted_std:.4f}")
print(f"Median of Actual Survival Time: {actual_median:.4f}")
print(f"Median of Predicted Survival Time: {predicted_median:.4f}")
print(f"Quartiles of Actual Survival Time: [{actual_q1:.4f}, {actual_q3:.4f}]")
print(f"Quartiles of Predicted Survival Time: [{predicted_q1:.4f}, {predicted_q3:.4f}]")
print(f"95% Confidence Interval of Actual Survival Time: [{actual_ci[0]:.4f}, {actual_ci[1]:.4f}]")
print(f"95% Confidence Interval of Predicted Survival Time: [{predicted_ci[0]:.4f}, {predicted_ci[1]:.4f}]")
print(f"Pearson Correlation Coefficient: {pearson_corr:.4f}")
print(f"Spearman Rank Correlation Coefficient: {spearman_corr:.4f}")
print(f"Kendall Rank Correlation Coefficient: {kendall_corr:.4f}")
print(f"Coefficient of Determination (R²): {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
