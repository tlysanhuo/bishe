import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimSun'
# 读取dat文件夹路径
data_folder = 'C:\\Users\\Admin\\Desktop\\TE化工数据集\\测试集'

# 创建保存图像和滤波后数据的文件夹
save_folder = 'C:\\Users\\Admin\\Desktop\\小波滤波图像'
os.makedirs(save_folder, exist_ok=True)

# 遍历每个dat文件
for filename in os.listdir(data_folder):
    if filename.endswith('.dat'):
        # 读取dat文件
        data = np.loadtxt(os.path.join(data_folder, filename))

        # 创建一个DataFrame用于保存滤波后的数据
        filtered_data = pd.DataFrame()

        # 遍历每一列数据
        for i in range(data.shape[1]):
            # 提取第i列数据
            column_data = data[:, i]

            # 如果该列数据全为零，则跳过
            if np.all(column_data == 0):
                continue

            # 使用小波滤波器进行滤波
            coeffs = pywt.wavedec(column_data, 'db4', level=3)
            coeffs[1:] = (pywt.threshold(c, value=0.5, mode='soft') for c in coeffs[1:])
            reconstructed_signal = pywt.waverec(coeffs, 'db4')

            # 将滤波后的数据保存到DataFrame中
            filtered_data[f'column_{i + 1}'] = reconstructed_signal

        # 创建保存当前dat文件图像和滤波后数据的文件夹
        save_folder_dat = os.path.join(save_folder, filename.split('_')[0])
        os.makedirs(save_folder_dat, exist_ok=True)

        # 将滤波后的数据保存为xlsx文件
        filtered_data.to_excel(os.path.join(save_folder_dat, f'{filename.split(".")[0]}.xlsx'), index=False)

        # 绘制前后对比图并保存在文件夹中
        for column in filtered_data.columns:
            plt.figure(figsize=(10, 5))

            # 绘制原始数据（红色）
            plt.plot(data[:, int(column.split('_')[1]) - 1], color='red', label='原始数据')

            # 绘制滤波后的数据（蓝色）
            plt.plot(filtered_data[column], color='blue', label='滤波后数据')

            # 设置图例
            plt.legend()

            # 设置坐标轴标签和标题
            plt.xlabel('样本点')
            plt.ylabel('数值')
            plt.title(f'变量{column.split("_")[1]}数据的前后对比')

            # 保存图像到文件夹中
            save_path = os.path.join(save_folder_dat, f'column_{column.split("_")[1]}_comparison.png')
            plt.savefig(save_path)

            # 关闭图像窗口
            plt.close()

print("图像和滤波后数据已保存在相应文件夹中。")