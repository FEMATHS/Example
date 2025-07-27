# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
import time
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

def f(t, u):
    return u - 2 * t / u

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
    elif Funi == 2:
        # 积分方法
        t_list = [i * ti for i in range(int(T) + 1)]

        for m in range(1, int(T) + 1):
            t_prev = t_list[m - 1]
            u_prev = U[m - 1]

            # 简单线性插值（只有两点）
            if m >= 2:
                interp_func = interp1d(t_list[:m], U[:m], kind='linear', fill_value="extrapolate")
                u_interp = lambda tau: float(interp_func(tau))
            else:
                u_interp = lambda tau: u_prev  # 初始步，常值延拓

            integrand = lambda tau: f(tau, u_interp(tau))
            integral, _ = quad(integrand, t_prev, t_prev + ti)

            U[m] = u_prev + integral
    elif Funi == 3:
        for m in range(1, int(T) + 1):
            t_prev = (m - 1) * ti
            u_prev = U[m - 1]

            # 一阶导数
            f_val = f(t_prev, u_prev)

            # 手动导数
            df_dt = -2 / u_prev
            df_du = 1 + (2 * t_prev) / (u_prev ** 2)

            # 二阶导数
            u2 = df_dt + df_du * f_val

            # 可以使用三阶的，这里只使用到二阶
            # 二阶泰勒展开
            U[m] = u_prev + ti * f_val + (ti ** 2) / 2 * u2

    return U


def UTrue_function(hi=1):
    ti = get_time_step(hi)
    T = 1 / ti
    U_T = [0] * (int(T) + 1)
    for m in range(0, int(T) + 1):
        U_T[m] = np.sqrt(1 + 2 * m * ti)
        # print("现在计算第", m * ti, "s的精确解为", U_T[m])
    return U_T

def Pre_error(hi=1):
    ti = get_time_step(hi)
    T = 1 / ti
    Pre_error = [0] * (int(T) + 1)
    for m in range(0, int(T) + 1):
        Pre_error[m] = 1/12 *((1+3/16)**(m) -1)
    return Pre_error

# 主程序
Funi = 1
u_0 = 1

U_N11 = example_function(1, 1, u_0)
U_N21 = example_function(2, 1, u_0)
U_N31 = example_function(3, 1, u_0)
U_N12 = example_function(1, 2, u_0)
U_N22 = example_function(2, 2, u_0)
U_N32 = example_function(3, 2, u_0)
U_N13 = example_function(1, 3, u_0)
U_N23 = example_function(2, 3, u_0)
U_N33 = example_function(3, 3, u_0)

U_T1 = UTrue_function(1)
U_T2 = UTrue_function(2)
U_T3 = UTrue_function(3)  # 用精度最高的那组作对比

error11 = np.abs(np.array(U_N11) - np.array(U_T1))
error21 = np.abs(np.array(U_N21) - np.array(U_T2))
error31 = np.abs(np.array(U_N31) - np.array(U_T3))
error12 = np.abs(np.array(U_N12) - np.array(U_T1))
error22 = np.abs(np.array(U_N22) - np.array(U_T2))
error32 = np.abs(np.array(U_N32) - np.array(U_T3))
error13 = np.abs(np.array(U_N13) - np.array(U_T1))
error23 = np.abs(np.array(U_N23) - np.array(U_T2))
error33 = np.abs(np.array(U_N33) - np.array(U_T3))

Pre_error1 = Pre_error(1)
Pre_error2 = Pre_error(2)
Pre_error3 = Pre_error(3)

# 统一长度
N = len(U_T3)
T = 1.0
# 为每组解分别生成时间向量
time1 = np.linspace(0, 1.0, len(U_T1))
time2 = np.linspace(0, 1.0, len(U_N21))
time3 = np.linspace(0, 1.0, len(U_N31))
time_true = np.linspace(0, 1.0, len(U_T3))


# === 误差曲线绘图 ===
plt.figure(1, figsize=(8, 6))

# 数值误差（Numerical Error）+ marker
plt.plot(time1, np.array(error12) + 1e-16, label='Numerical Error (2⁴)', linestyle='-', marker='o', markersize=4, color='blue')
plt.plot(time2, np.array(error22) + 1e-16, label='Numerical Error (2⁸)', linestyle='-', markersize=4, color='green')
plt.plot(time3, np.array(error32) + 1e-16, label='Numerical Error (2¹⁰)', linestyle='-', markersize=4, color='red')

# 先验误差（Prior Error）+ marker
plt.plot(time1, np.array(Pre_error1) + 1e-16, label='Prior Error (2⁴)', linestyle='--', marker='o', markersize=4, color='blue')
plt.plot(time2, np.array(Pre_error2) + 1e-16, label='Prior Error (2⁸)', linestyle='--', markersize=4, color='green')
plt.plot(time3, np.array(Pre_error3) + 1e-16, label='Prior Error (2¹⁰)', linestyle='--', markersize=4, color='red')

# 图形设置
plt.xlabel('Time (s)')
plt.ylabel('Error')
plt.title('Comparison of Numerical Error and Prior Error')
plt.yscale('log')  # 使用对数坐标可更清楚展示误差衰减趋势
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.legend()
plt.tight_layout()
plt.savefig('1.png', dpi=300)


plt.figure(2, figsize=(8, 6))
plt.plot(time1, np.array(Pre_error1) / np.array(error12), label='Prior / Numerical (2⁴)')
plt.plot(time2, np.array(Pre_error2) / np.array(error22), label='Prior / Numerical (2⁸)')
plt.plot(time3, np.array(Pre_error3) / np.array(error32), label='Prior / Numerical (2¹⁰)')
plt.axhline(1, color='gray', linestyle='--', linewidth=1)
plt.xlabel('Time (s)')
plt.ylabel('Prior / Numerical Error')
plt.yscale('log')  # 使用对数坐标可更清楚展示误差衰减趋势
plt.title('Ratio Between Prior and Numerical Error')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('2.png', dpi=300)
plt.show()
