'''import math
# 导入需要的库
import json
import random

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import seaborn as sns

# 下面是做计算了
# 读取json数据
with open('results.json') as f:
    result = json.load(f)

with open('eyesuccess.json') as f:
    surgery_num = json.load(f)

# 提取特征
x = []
y = []
y_mean = 0
count = 0
y_sum_squares = 0

for id, name1 in result.items():
    if surgery_num.get(id, 0) == 3 and name1['mean'] < 0.99999:
        x.append(surgery_num.get(id, 0))
        y.append(name1['mean'])
        y_mean += name1['mean']
        y_sum_squares += name1['mean'] ** 2
        count += 1

y_std = math.sqrt(y_sum_squares/count - (y_mean/count)**2)
print(y_mean/count)
print(count)
print(y_std)


'''
import math
# 导入需要的库
import json
import random

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score

# 下面是做计算了
# 读取json数据
with open('results.json') as f:
    result = json.load(f)

with open('eyesuccess.json') as f:
    surgery_num = json.load(f)

# 提取特征
x = []
y = []
y_mean = 0
count = 0
y_sum_squares = 0

for id, name1 in result.items():
    if surgery_num.get(id, 0) == 3 and name1['mean'] < 0.99999:
        x.append(surgery_num.get(id, 0))
        y.append(name1['mean'])
        y_mean += name1['mean']
        y_sum_squares += name1['mean'] ** 2
        count += 1

y_std = math.sqrt(y_sum_squares/count - (y_mean/count)**2)
print(y_mean/count)
print(count)
print(y_std)

