# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False
# ✅ 数学字体设置为 Times New Roman（关键点）
plt.rcParams['mathtext.fontset'] = 'custom'               # 自定义字体方案
plt.rcParams['mathtext.rm'] = 'Times New Roman'           # 普通公式字体
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'    # 斜体公式字体
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'      # 粗体公式字体

# 统一字体大小（适合高分辨率绘图）
plt.rcParams['font.size'] = 24               # 所有字体默认大小
plt.rcParams['axes.titlesize'] = 24          # 坐标轴标题
plt.rcParams['axes.labelsize'] = 24          # x/y/z轴标签
plt.rcParams['xtick.labelsize'] = 20         # x轴刻度
plt.rcParams['ytick.labelsize'] = 20         # y轴刻度
plt.rcParams['legend.fontsize'] = 20         # 图例字体
plt.rcParams['figure.titlesize'] = 24        # 图标题字体
def get_time_step(hi):
    if hi == 1:
        return 1 / (2 ** 4)
    elif hi == 2:
        return 1 / (2 ** 8)
    elif hi == 3:
        return 1 / (2 ** 10)
    else:
        raise ValueError("hi must be 1, 2, or 3")

def example_function(hi=1, Funi=1, u_0=1):
    assert u_0 != 0, "初始值 u_0 不能为 0，否则会除以 0 崩溃"
    ti = get_time_step(hi)
    T = 1 / ti
    U = [0] * (int(T) + 1)
    U[0] = u_0
    if Funi == 1:
        for m in range(1, int(T) + 1):
            U[m] = U[m - 1] + (U[m - 1] - (2 * m * ti) / (U[m - 1])) * ti
            # print("现在计算第", m * ti, "s的数值解为", U[m])
    return U

def UTrue_function(hi=1):
    ti = get_time_step(hi)
    T = 1 / ti
    U_T = [0] * (int(T) + 1)
    for m in range(0, int(T) + 1):
        U_T[m] = np.sqrt(1 + 2 * m * ti)
        # print("现在计算第", m * ti, "s的精确解为", U_T[m])
    return U_T

# 主程序
Funi = 1
u_0 = 1

U_N1 = example_function(1, Funi, u_0)
U_N2 = example_function(2, Funi, u_0)
U_N3 = example_function(3, Funi, u_0)
U_T1 = UTrue_function(1)
U_T2 = UTrue_function(2)
U_T3 = UTrue_function(3)  # 用精度最高的那组作对比

error1 = np.abs(np.array(U_N1) - np.array(U_T1))
error2 = np.abs(np.array(U_N2) - np.array(U_T2))
error3 = np.abs(np.array(U_N3) - np.array(U_T3))

# 统一长度
N = len(U_T3)
T = 1.0
time = np.linspace(0, T, N)

# 为每组解分别生成时间向量
time1 = np.linspace(0, 1.0, len(U_T1))
time2 = np.linspace(0, 1.0, len(U_N2))
time3 = np.linspace(0, 1.0, len(U_N3))
time_true = np.linspace(0, 1.0, len(U_T3))

# 分别绘图
plt.figure(1, figsize=(8, 6))  
plt.plot(time1, U_N1, label='Numerical Solution 1 (2^4)',marker='o', markersize=5)
plt.plot(time2, U_N2, label='Numerical Solution 2 (2^8)')
plt.plot(time3, U_N3, label='Numerical Solution 3 (2^10)')
plt.plot(time_true, U_T3, label='Exact Solution', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('U(t)')
plt.title('Comparison of Numerical and Exact Solutions')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('1.png', dpi=300)  # 保存图像

plt.figure(2, figsize=(8, 6))  
plt.plot(time1, error1 + 1e-16, label='Error 1 (2^4)')
plt.plot(time2, error2, label='Error 2 (2^8)')
plt.plot(time3, error3, label='Error 3 (2^10)')

plt.xlabel('Time (s)')
plt.ylabel('Absolute Error')
plt.title('Absolute Error between Numerical and Exact Solutions')
plt.legend()
plt.grid(True, which='both')  # 网格显示，both 使得主次网格都显示
plt.yscale('log')  # 关键：y轴对数刻度
plt.tight_layout()
plt.savefig('2.png', dpi=300)  # 保存图像
plt.show()
