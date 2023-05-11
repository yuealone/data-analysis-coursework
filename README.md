# data-analysis-coursework
1、上传数据
from google.colab import files
uploaded = files.upload()

2、导入Pandas库并使用read_csv函数读取.csv文件
import pandas as pd
df = pd.read_csv('dataset_039.csv')

3、检查数据的基本信息，包括列名、数据类型、空值情况等
df.info()

4、检查缺失值（本数据没有）
print(df.isnull().sum())

5、检查异常值

5.1首先，可以通过 describe() 函数来查看每一列的统计信息，包括均值、标准差、最小值、最大值、中位数等等。
print(df.describe())

5.2然后，可以检查每一列是否存在异常值。可以使用箱线图和直方图等方式来检查异常值。例如，使用箱线图可以检查数值型数据中是否存在离群点。这段代码会画出 age 这一列的箱线图。如果有离群点，可以考虑将其删除或替换
import matplotlib.pyplot as plt

# 画箱线图
plt.boxplot(df['age'])
plt.show()

5.2然后，可以检查每一列是否存在异常值。可以使用箱线图和直方图等方式来检查异常值。例如，使用箱线图可以检查数值型数据中是否存在离群点。这段代码会画出 age 这一列的箱线图。如果有离群点，可以考虑将其删除或替换

5.3使用 Z-score 方法检测异常值

5.3.1原始数据
print(df.head())


5.3.2处理后的数据
num_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
# 设置 Z-score 阈值为 3
threshold = 3
# 找到所有 Z-score 绝对值大于阈值的数据点
import numpy as np
outliers = np.abs(df[num_cols]) > threshold
# 将异常值替换为 NA
df[outliers] = np.nan
# 删除所有包含 NA 的行
df = df.dropna()
# 查看处理后的数据
print(df.head())
#1、数据变化很大是因为在进行Z-score标准化处理时，对数据进行了均值和标准差的转换。
#2、数据集中的每个变量都被标准化为均值为0，标准差为1。这是在检测并处理异常值后执行的。因此，可以认为在第二个数据集中，异常值已被处理，数据已进行标准化处理。
#3、在使用Z-score方法进行异常值检测时，任何超过阈值的数据点都被认为是异常值并进行替换。因此，如果存在任何异常值，它们会被检测到并被替换为平均值或中位数，这就是为什么你在数据经过处理后，值的范围被压缩到了一个较小的区间内
#这也意味着可能存在异常值被处理了。但是，这个过程并不能精确地确定每个异常值的位置，因为所有超过阈值的值都会被替换。如果你需要更精确的异常值检测，可以尝试使用其他方法，例如IQR或Z-score方法的改进版本）
处理后的数据


6、标准化数据：自定义映射（Custom Mapping）
education_mapping = {'secondary': 2, 'primary': 1, 'unknown': 4, 'tertiary': 3}
job_mapping = {'technician': 1, 'entrepreneur': 2, 'admin.': 3, 'retired': 4,'blue-collar': 5, 'management': 6, 'self-employed': 7, 'services': 8,'unemployed': 9, 'housemaid': 10, 'student': 11, 'unknown': 12}
marital_mapping = {'single': 1, 'married': 2, 'divorced': 3}
default_mapping = {'no': -1, 'yes': 1}
housing_mapping = {'no': -1, 'yes': 1}
loan_mapping = {'no': -1, 'yes': 1}
contact_mapping = {'telephone': 1, 'cellular': 2, 'unknown': 3}
month_mapping = {'jan': 1, 'feb': 2, 'mar.': 3, 'apr': 4,'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
poutcome_mapping = {'success': 1, 'failure': 2, 'other': 3, 'unknown': 4}
y_mapping = {'yes': 1, 'no': -1}

# 使用map函数进行自定义映射
df['education_mapped'] = df['education'].map(education_mapping)
df['job_mapped'] = df['job'].map(job_mapping)
df['marital_mapped'] = df['marital'].map(marital_mapping)
df['default_mapped'] = df['default'].map(default_mapping)
df['housing_mapped'] = df['housing'].map(housing_mapping)
df['loan_mapped'] = df['loan'].map(loan_mapping)
df['contact_mapped'] = df['contact'].map(contact_mapping)
df['month_mapped'] = df['month'].map(month_mapping)
df['poutcome_mapped'] = df['poutcome'].map(poutcome_mapping)
df['y_mapped'] = df['y'].map(y_mapping)



6.1、发现education等列的文本数据已经转换成了数字，就在mapped那一列，后面只要将使用转换后的数值列作为输入特征，进行训练和预测就可以。遗留的文本列本身不会直接影响模型的性能指标。
print(df.head())

6.1(补)如果转换失败,可以改变数据类型（因为自定义映射只能转换字符串类型（object）数据，先观察再进行转换）
#观察数据类型
print(df['education'].dtypes)
#转换数据类型
df['education'] = df['education'].astype(str)
