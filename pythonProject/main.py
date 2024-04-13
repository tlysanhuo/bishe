import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pykalman import KalmanFilter

# 读取dat文件
data = np.loadtxt('C:\\Users\\Admin\\Desktop\\TE化工数据集\\测试集\\d00_te.dat')

# 创建一个DataFrame用于保存滤波后的数据
filtered_data = pd.DataFrame()

# 遍历每一列数据
for i in range(data.shape[1]):
    # 提取第i列数据
    column_data = data[:, i]

    # 如果该列数据全为零，则跳过
    if np.all(column_data == 0):
        continue

    # 创建卡尔曼滤波器对象
    kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])

    # 使用卡尔曼滤波器进行滤波
    filtered_state_means, filtered_state_covariances = kf.filter(column_data)

    # 获取滤波后的结果
    filtered_signal = filtered_state_means.flatten()

    # 将滤波后的数据保存到DataFrame中
    filtered_data[f'column_{i + 1}'] = filtered_signal

# 将滤波后的数据保存为xlsx文件
filtered_data.to_excel('C:\\Users\\Admin\\Desktop\\卡尔曼滤波.xlsx', index=False)

# 创建保存图像的文件夹
save_folder = 'C:\\Users\\Admin\\Desktop\\卡尔曼滤波'
os.makedirs(save_folder, exist_ok=True)

# 绘制前后对比图并保存在文件夹中
for column in filtered_data.columns:
    plt.figure(figsize=(10, 5))

    # 绘制原始数据（红色）
    plt.plot(data[:, int(column.split('_')[1]) - 1], color='red', label='原始数据')

    # 绘制滤波后的数据（蓝色）
    plt.plot(filtered_data[column], color='blue', label='滤波后的数据')

    # 设置图例
    plt.legend()

    # 设置坐标轴标签和标题
    plt.xlabel('样本点')
    plt.ylabel('数值')
    plt.title(f'列{column.split("_")[1]}数据的前后对比')

    # 保存图像到文件夹中
    save_path = os.path.join(save_folder, f'column_{column.split("_")[1]}_comparison.png')
    plt.savefig(save_path)

    # 关闭图像窗口
    plt.close()

print("图像已保存在文件夹中。")