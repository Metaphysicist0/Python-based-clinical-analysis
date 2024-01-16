import math
# 导入需要的库
import json
import random

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import seaborn as sns

e_file = '/.json'
re_file = '/.json'
output_file = '.json'

with open(e_file, 'r') as f:
    e_data = json.load(f)

with open(r_file, 'r') as f:
    r_data = json.load(f)

ratio_data = {}

for name in :
    e_value = e_data[name]
    r_value = r_data[name]

    ratio = (r_value / e_value)*10
    # sigmoid归一化
    # normalized_
    # ratio_data[name] = normalized_ratio

    ratio_data[name] = math.tanh(ratio)


with open(output_file, 'w') as f:
    json.dump(ratio_data, f, indent=4)   # 增加缩进

with open('.json') as f:
    data = json.load(f)

with open('.json') as f:
    surgery_num = json.load(f)

results = {}

for key, value in data.items():
    identifier = key.split('_')[0]
    if identifier not in results:
        results[identifier] = {}

count = 0  # 定义一个计数器
for identifier, result in results.items():
    value_list = [value for key, value in data.items() if key.startswith(identifier)]
    num = surgery_num.get(identifier, 0)
    # mean = sum(value_list) / len(value_list)  # 修改前的代码

    max_value = max(value_list)
    min_value = min(value_list)

    if len(value_list) > 2:
        value_list.remove(max_value)  # 去掉最大值
        value_list.remove(min_value)  # 去掉最小值

    if num == 0:
        mean = 0
    else:
        mean = sum(value_list) / num

    var = sum((x - mean) ** 2 for x in value_list) / (num+1)
    std = var ** 0.5

    results[identifier]['mean'] = mean
    results[identifier]['var'] = var
    results[identifier]['std'] = std

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

# 下面是做计算了
# 读取json数据
with open('results.json') as f:
    result = json.load(f)

with open('.json') as f:
    s_num = json.load(f)

# 提取特征
x = []
y = []

for id, name1 in result.items():
    x.append(s_num.get(id, 0))
    y.append(name1['mean'])

# 计算皮尔逊相关系数
corr, p1 = pearsonr(x, y)
rho, p2 = spearmanr(x, y)

print("Pearson Correlation: ", corr)
print("P Value: ", p1)
print("Spearman Rank Correlation: ", rho)
print("P Value: ", p2)


# 定义一个函数，用来检查数据是否满足斯皮尔曼相关系数的条件


def check_spearman(x, y):
    # 首先检查两个变量是否是有序变量或者连续变量
    from scipy.stats import rankdata
    x_rank = rankdata(x)
    y_rank = rankdata(y)
    if (x_rank == x).all() or (y_rank == y).all():
        print("One of the variables is ordinal.")
    else:
        print("Both variables are continuous.")
    # 然后检查两个变量是否呈单调关系
    from scipy.stats import kendalltau
    tau, p = kendalltau(x, y)
    if p < 0.05:
        print("The variables have a significant monotonic relationship.")
    else:
        print("The variables do not have a significant monotonic relationship.")

# 使用自定义的函数检查数据是否满足斯皮尔曼相关系数的条件
check_spearman(x, y)

THRESHOLD = 0.15
stable_count = 0
with open('results.json') as f:
    results = json.load(f)

for id, data in results.items():
    var = data['var']
    std = data['std']

    total_count = len(results)

    if var < THRESHOLD:
        stable_count += 1

print(f"Total samples: {total_count}")
print(f"Stable samples: {stable_count}")
print(f"Percentage stable: {stable_count / total_count * 100:.2f}%")

import statistics

results_by_id = {}
nums_by_group = {}

for id, data in results.items():
    if id not in results_by_id:
        results_by_id[id] = []
    results_by_id[id].append(data['mean'])

with open('surgerynumber.json') as f:
    surgery_nums = json.load(f)


for id, means in results_by_id.items():
    num = surgery_nums[id]
    if num not in nums_by_group:
        nums_by_group[num] = []
    nums_by_group[num].extend(means)

nums = list(nums_by_group.keys())
nums.sort()

for num in nums:
    means = nums_by_group[num]
    means = [mean for mean in means if mean < 1]
    avg = statistics.mean(means)
    std = statistics.pstdev(means)
    print(f"{avg},{std},{num}")

import json
import pandas as pd
import seaborn as sns

with open('.json') as f:
    surgery_nums = json.load(f)

with open('results.json') as f:
    results = json.load(f)

df = pd.DataFrame(columns=[' ', 'mean RPR'])

for id, data in results.items():
    surgery_num = surgery_nums[id]
    mean = data['mean']
    if mean < 0.99999:
        df = df.append({' ': surgery_num,
                            'mean RPR': mean}, ignore_index=True)
g = sns.catplot(x=" ", y="mean RPR", data=df, kind="box")
g.set_titles("{col_name}")

plt.show()


groups = df[' '].unique()
means = []
stds = []
for g in groups:
    mean = df[df[' '] == g]['mean RPR'].mean()
    means.append(mean)

g = sns.catplot(x=" ", y="mean RPR", data=df, kind="box")

import json
import pandas as pd
import seaborn as sns

with open('.json') as f:
    surgery_nums = json.load(f)

with open('results.json') as f:
    results = json.load(f)

df = pd.DataFrame(columns=[' ', 'mean RPR'])

num_dict = {}

for id, data in results.items():
    surgery_num = surgery_nums[id]
    mean = data['mean']
    if mean < 0.99999:
        df = df.append({' ': surgery_num,
                        'mean RPR': mean}, ignore_index=True)

        if surgery_num not in num_dict:
            num_dict[surgery_num] = 0
        num_dict[surgery_num] += 1

print(":")
print(num_dict)

g = sns.catplot(x=" ", y="mean RPR", data=df, kind="box")
g.set_titles("{col_name}")

plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# 画热力图
corr = df.corr()

fig, ax = plt.subplots(figsize=(10, 8))
ax = sns.heatmap(corr,
                 xticklabels=corr.columns,
                 yticklabels=corr.columns)

# 获取每个单元格的相关系数值
for i in range(len(corr)):
    for j in range(len(corr.columns)):
        text = ax.text(j+0.5, i+0.5, round(corr.iloc[i,j],2),
                       ha="center", va="center", color="w")

# 设置轴标签字体大小
ax.set_xticklabels(ax.get_xticklabels(), size=11)
ax.set_yticklabels(ax.get_yticklabels(), size=11)

plt.show()

