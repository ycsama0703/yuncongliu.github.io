from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Loading the data
dat = pd.read_csv("2019-06-13-exam-pa-data-file.csv")

# Variables to relevel
vars = dat.columns[4:14]  # Adjusted indexing for Python

for var in vars:
    # Compute frequency table for the variable
    table = dat[var].value_counts().reset_index()
    table.columns = ['Level', 'Count']
    
    # Find the level with the maximum count
    max_level = table.loc[table['Count'].idxmax(), 'Level']
    
    # Reorder levels, moving the most frequent level to the front
    # Since pandas does not have a direct equivalent of factor releveling as in R,
    # we simulate it by sorting based on whether the value is the max_level or not
    dat[var] = pd.Categorical(dat[var], categories=[max_level] + [x for x in dat[var].unique() if x != max_level], ordered=True)


import matplotlib.pyplot as plt

# Plotting histogram of Crash_Score
plt.hist(dat['Crash_Score'], bins='auto', color='blue', alpha=0.7)
plt.xlabel('Crash_Score')
plt.ylabel('Frequency')
plt.title('Histogram of Crash_Score')
plt.show()



import seaborn as sns


vars = [col for col in dat.columns if col != "Crash_Score"]  # Exclude "Crash_Score"

for var in vars:
    # Convert the column to a categorical type if it's not already
    dat[var] = dat[var].astype('category')
    
    # Create a boxplot
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    sns.boxplot(x=var, y=np.log(dat['Crash_Score']), data=dat)
    plt.xlabel(var)
    plt.ylabel('Log of Crash_Score')
    plt.title(f'Box Plot of Log of Crash_Score vs {var}')
    plt.xticks(rotation=45)  # Rotate labels to avoid overlap
    plt.show()




for var in vars:
    #print(var)
    grouped = dat.groupby(var).agg(
        mean_log_Crash_Score=('Crash_Score', lambda x: np.mean(np.log(x))),
        median_log_Crash_Score=('Crash_Score', lambda x: np.median(np.log(x))),
        n=('Crash_Score', 'size')
    ).reset_index()
    #print(grouped)



# Reduce the number of factor levels where appropriate

def draw_bar(col_name):
    plt.figure(figsize=(10, 6))  # 设置图形的大小
    sns.countplot(x='{}'.format(col_name), data=dat2, order = dat2[col_name].value_counts().index)
    plt.title('Frequency of Each Category in {}'.format(col_name))
    plt.show()

for var in vars:
    # Setting the plot size for better readability
    plt.figure(figsize=(10, 6))
    
    # Count plot for each variable
    sns.countplot(x=var, data=dat, palette="viridis")
    
    # Rotating x-axis labels for better readability
    plt.xticks(rotation=90)
    
    plt.xlabel(var)  # Setting the x-axis label to the current variable
    plt.ylabel('Count')  # Setting the y-axis label
    plt.title(f'Bar Plot of {var}')  # Setting the title of the plot
    
    # Show plot
    plt.show()







# Convert 'Time_of_Day' to a categorical type
dat['Time_of_Day'] = dat['Time_of_Day'].astype('category')

# Print the original levels (categories)
print(dat['Time_of_Day'].cat.categories)

# Create a copy of the DataFrame for modifications
dat2 = dat.copy()

# Manually adjust the levels for 'Time_of_Day'
replacement_mapping = {
    1: "OVERNIGHT",
    2: "LATE-EARLY",
    # Assuming "Original_Level_3" and "Original_Level_4" map to "DAYTIME",
    # and so on. Replace these keys with the actual original levels from your dataset.
    3: "DAYTIME",
    4: "DAYTIME",
    5: "DAYTIME",
    6: "LATE-EARLY"
}

# Replace the levels based on the mapping
dat2['Time_of_Day'] = dat2['Time_of_Day'].replace(replacement_mapping)

# Convert again to categorical if needed and relevel based on frequency
dat2['Time_of_Day'] = dat2['Time_of_Day'].astype('category')
most_common_level = dat2['Time_of_Day'].value_counts().idxmax()

# Set the most common level as the first category
dat2['Time_of_Day'] = dat2['Time_of_Day'].cat.reorder_categories([most_common_level] + [cat for cat in dat2['Time_of_Day'].cat.categories if cat != most_common_level], ordered=True)

# Display the frequency table for the modified 'Time_of_Day'
draw_bar('Time_of_Day')



# Convert 'Rd_Feature' to a categorical type if not already
dat['Rd_Feature'] = dat['Rd_Feature'].astype('category')

# Print the original levels (categories)
print("Original levels in 'Rd_Feature':", dat['Rd_Feature'].cat.categories)


# Define the mapping for the category adjustments
# Assuming you know the original levels, replace them accordingly in the dictionary below
replacement_mapping_for_rd_feature = {
    "NONE": "OTHER",
    "DRIVEWAY": "OTHER",
    "INTERSECTION": "INTERSECTION",
    "RAMP": "OTHER",
    "OTHER": "OTHER",
    # Add or remove mappings based on your actual data
}

# Apply the replacement mapping
dat2['Rd_Feature'] = dat2['Rd_Feature'].replace(replacement_mapping_for_rd_feature).astype('category')

# Find the most common level (category) in the 'Rd_Feature'
most_common_level = dat2['Rd_Feature'].value_counts().idxmax()

# Reorder the levels so that the most common level comes first
dat2['Rd_Feature'] = dat2['Rd_Feature'].cat.reorder_categories([most_common_level] + [cat for cat in dat2['Rd_Feature'].cat.categories if cat != most_common_level], ordered=True)

# Display the frequency table for the 'Rd_Feature' column to verify the changes
draw_bar('Rd_Feature')



dat2['Rd_Character'] = dat2['Rd_Character'].astype('category')

# Replace specified levels with new designations
replacement_mapping_rd_character = {
    "STRAIGHT-LEVEL": "STRAIGHT",
    "CURVE-LEVEL": "CURVE",
    # Adjust the original levels and mappings as necessary
    "CURVE-GRADE": "CURVE",
    "CURVE-OTHER": "CURVE",
    "OTHER": "CURVE",
    "STRAIGHT-GRADE": "STRAIGHT",
    "STRAIGHT-OTHER": "STRAIGHT"
}
dat2['Rd_Character'] = dat2['Rd_Character'].replace(replacement_mapping_rd_character).astype('category')

# Relevel based on the frequency
most_common_level_rd_character = dat2['Rd_Character'].value_counts().idxmax()
dat2['Rd_Character'] = dat2['Rd_Character'].cat.reorder_categories([most_common_level_rd_character] + [cat for cat in dat2['Rd_Character'].cat.categories if cat != most_common_level_rd_character], ordered=True)

# Display the frequency table for 'Rd_Character'
draw_bar('Rd_Character')



dat2['Rd_Surface'] = dat2['Rd_Surface'].astype('category')

# 定义一个映射来合并等级为 'ASPHALT' 和 'OTHER'
replacement_mapping_for_Rd_Surface = {
    "SMOOTH ASPHALT": "ASPHALT",
    "COARSE ASPHALT": "ASPHALT",
    "CONCRETE": "OTHER",
    "GROOVED CONCRETE": "OTHER",
    "OTHER": "OTHER"
}

# 应用映射
dat2['Rd_Surface'] = dat2['Rd_Surface'].replace(replacement_mapping_for_Rd_Surface)

# 重新转换为分类类型以确保数据类型，并为可能的重新排序做准备
dat2['Rd_Surface'] = dat2['Rd_Surface'].astype('category')

# 确定最常见的等级
most_common_surface = dat2['Rd_Surface'].value_counts().idxmax()

# 重新排序等级，将最常见的等级放在首位
dat2['Rd_Surface'] = dat2['Rd_Surface'].cat.reorder_categories(
    [most_common_surface] + [cat for cat in dat2['Rd_Surface'].cat.categories if cat != most_common_surface],
    ordered=True
)

# 显示修改后的等级及其频率
draw_bar('Rd_Surface')



dat2['Traffic_Control'] = dat2['Traffic_Control'].astype('category')

# Print the original levels for 'Traffic_Control'
print("Original Levels:", dat2['Traffic_Control'].cat.categories)

# Define a mapping for consolidating levels into 'SIGNAL-STOP' and 'OTHER'
replacement_mapping_for_Traffic_Control = {
    "NONE": "OTHER",
    "YIELD": "OTHER",
    "SIGNAL": "SIGNAL-STOP",
    "STOP-SIGN": "SIGNAL-STOP",
    "OTHER": "OTHER"
    # Adjust the keys according to the actual original levels of 'Traffic_Control'
}

# Apply the mapping
dat2['Traffic_Control'] = dat2['Traffic_Control'].replace(replacement_mapping_for_Traffic_Control)

# Re-convert to categorical to enforce the data type and for possible reordering
dat2['Traffic_Control'] = dat2['Traffic_Control'].astype('category')

# Identify the most common level
most_common_traffic_control = dat2['Traffic_Control'].value_counts().idxmax()

# Reorder the levels, placing the most common level first
dat2['Traffic_Control'] = dat2['Traffic_Control'].cat.reorder_categories(
    [most_common_traffic_control] + [cat for cat in dat2['Traffic_Control'].cat.categories if cat != most_common_traffic_control], 
    ordered=True
)

# Display the modified levels and their frequencies
draw_bar('Traffic_Control')



from sklearn.preprocessing import OneHotEncoder

# 使用 sparse_output 替代 sparse
encoder = OneHotEncoder(sparse_output=False, drop=None)  # 保留所有特征，适用于 PCA

# 剩余的代码不变
dat_pca_bin = encoder.fit_transform(dat2)
dat_pca_bin_df = pd.DataFrame(dat_pca_bin, columns=encoder.get_feature_names_out())

# 显示前几行以验证结果
print(dat_pca_bin_df.head())


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 首先，对数据进行标准化
scaler = StandardScaler()
dat_pca_scaled = scaler.fit_transform(dat_pca_bin_df)

# 执行 PCA
pca_weather = PCA()
pca_weather.fit(dat_pca_scaled)

# 打印 PCA 分析的总结信息
print("解释的方差比率（每个主成分）:")
print(pca_weather.explained_variance_ratio_)

print("\n主成分的旋转（载荷）:")
# 载荷可以通过 components_ 属性获得。为了更好地理解，我们可以将其转换为 DataFrame
components_df = pd.DataFrame(pca_weather.components_, columns=dat_pca_bin_df.columns)
print(components_df)

# 如果你想查看特定数量的主成分，比如前几个，可以在 PCA() 中设置 n_components 参数
# 例如：pca_weather = PCA(n_components=5)

#解释的方差比率的条形图
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca_weather.explained_variance_ratio_) + 1), pca_weather.explained_variance_ratio_)
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')
plt.title('PCA Explained Variance Ratio')
plt.show()

#累积解释的方差比率
cumulative_variance_ratio = np.cumsum(pca_weather.explained_variance_ratio_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xlabel('Number of Principal Components')
plt.title('Cumulative PCA Explained Variance Ratio')
plt.show()

#主成分载荷（主成分与原始变量的相关性）
n_components_to_plot = 5  # 可以调整这个数值
components_to_plot = components_df.iloc[:n_components_to_plot, :]

plt.figure(figsize=(12, 8))
sns.heatmap(components_to_plot, cmap='viridis', annot=True, fmt=".2f")
plt.ylabel('Principal Components')
plt.xlabel('Features')
plt.title('Principal Component Loadings')
plt.xticks(rotation=90)  # 如果特征名称太长，可以旋转以便阅读
plt.show()




scaler = StandardScaler()
dat_pca_bin_std = scaler.fit_transform(dat_pca_bin_df)
dat_pca_bin_std_df = pd.DataFrame(dat_pca_bin_std, columns=dat_pca_bin_df.columns)

# 基于 PCA 结果创建新特征
# 这里使用的系数是基于你提供的 R 代码示例
dat2['WETorDRY'] = (-0.51 * dat_pca_bin_std_df['Rd_Conditions_DRY'] +
                    0.5 * dat_pca_bin_std_df['Rd_Conditions_WET'] -
                    0.46 * dat_pca_bin_std_df['Weather_CLEAR'] +
                    0.43 * dat_pca_bin_std_df['Weather_RAIN'])

print(dat2['WETorDRY'].describe())



# 以 'Rd_Conditions' 和 'Weather' 开头的列可能已经在之前的步骤中用于创建新特征
# 删除这些列的示例
for col in dat2.columns:
    if col.startswith('Rd_Conditions') or col.startswith('Weather'):
        del dat2[col]

# 检查数据以确认删除
print(dat2.head())


# 设置绘图风格
sns.set(style="whitegrid")

# 绘制箱形图
# 注意：我们需要先确保 Crash_Score 没有负数或零，因为对这些值取对数是无意义的
# 这里我们假设 dat2 中的 Crash_Score 都是正数
plt.figure(figsize=(12, 8))  # 设置图形大小
ax = sns.boxplot(x='Rd_Character', y=np.log(dat2['Crash_Score']), hue='Rd_Class', data=dat2)

# 设置图表标题和轴标签
plt.title('Boxplot of log(Crash_Score) by Rd_Character with Rd_Class')
plt.xlabel('Road Character')
plt.ylabel('Log of Crash Score')

# 由于 'Rd_Character' 的不同值可能很多，旋转 x 轴标签以改善可读性
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.tight_layout()  # 自动调整子图参数, 使之填充整个图像区域
plt.show()



# 首先为 'Crash_Score' 创建一个对数变换的列
dat2['Log_Crash_Score'] = np.log(dat2['Crash_Score'])

# 使用 FacetGrid 创建一个分面网格，每个 'Traffic_Control' 一个分面
g = sns.FacetGrid(dat2, col="Traffic_Control", col_wrap=4, height=4, sharex=False, sharey=False)

# 使用 map_dataframe 方法绘制箱形图，这里传递列名作为参数
g.map_dataframe(sns.boxplot, x='Rd_Feature', y='Log_Crash_Score', hue='Rd_Feature', palette='Set2')

# 设置每个分面的标题
g.set_titles(col_template="{col_name}", row_template="{row_name}")

# 添加图例
g.add_legend()

plt.show()


# 创建新的交叉项特征，并将其转换为字符串
dat2['Feature_Interaction'] = (dat2['Rd_Feature'].astype(str) + '_' + dat2['Traffic_Control'].astype(str))

# 将新特征转换为'category'类型
dat2['Feature_Interaction'] = dat2['Feature_Interaction'].astype('category')

# 显示结果以验证新列的类型
print(dat2[['Rd_Feature', 'Traffic_Control', 'Feature_Interaction']].head())
print(dat2['Feature_Interaction'].dtype)











# 假设'Feature_Interaction'是已经创建好的交叉项特征列
# 绘制条形图以显示每个交叉项的计数
sns.countplot(y='Feature_Interaction', data=dat2, order=dat2['Feature_Interaction'].value_counts().index)

# 为图表添加标题和标签
plt.title('Counts of Feature Interactions')
plt.xlabel('Count')
plt.ylabel('Feature Interaction')

# 显示图表
plt.show()





dat2.to_csv('Data_after_PCA_and_inter.csv', index=False)

dat2.describe(include=['category'])