import scipy.stats as stats
import numpy as np

mean1, std1 = 
mean2, std2 = 

n1 = 
n2 =

t_stat, p_value = stats.ttest_ind(a=np.random.normal(mean1, std1, n1),
b=np.random.normal(mean2, std2, n2),
equal_var=True)

print("进行两个平均值相等性独立样本T检验:")
print("t统计量是:", t_stat)
print("显著性差异p值是:", p_value)
