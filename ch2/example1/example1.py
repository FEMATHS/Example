# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
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

# 统一长度
N = len(U_T3)
T = 1.0
# 为每组解分别生成时间向量
time1 = np.linspace(0, 1.0, len(U_T1))
time2 = np.linspace(0, 1.0, len(U_N21))
time3 = np.linspace(0, 1.0, len(U_N31))
time_true = np.linspace(0, 1.0, len(U_T3))

# 分别绘图
plt.figure(1, figsize=(8, 6))  
plt.plot(time1, U_N11, label='Numerical Solution Fun1 (2^4)',marker='o', markersize=5)
plt.plot(time2, U_N21, label='Numerical Solution Fun1 (2^8)')
plt.plot(time3, U_N31, label='Numerical Solution Fun1 (2^10)')
plt.plot(time_true, U_T3, label='Exact Solution', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('U(t)')
plt.title('Numerical–Exact Solution Comparison (Fun1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('1.png', dpi=300)  # 保存图像

plt.figure(2, figsize=(8, 6))  
plt.plot(time1, error11 + 1e-16, label='Error 1 (2^4)')
plt.plot(time2, error21, label='Error 2 (2^8)')
plt.plot(time3, error31, label='Error 3 (2^10)')

plt.xlabel('Time (s)')
plt.ylabel('Absolute Error')
plt.title('Absolute Error (Fun1)')
plt.legend()
plt.grid(True, which='both')  # 网格显示，both 使得主次网格都显示
plt.yscale('log')  # 关键：y轴对数刻度
plt.tight_layout()
plt.savefig('2.png', dpi=300)  # 保存图像

# 分别绘图
plt.figure(3, figsize=(8, 6))  
plt.plot(time1, U_N11, label='Numerical Solution Fun2 (2^4)',marker='o', markersize=5)
plt.plot(time2, U_N21, label='Numerical Solution Fun2 (2^8)')
plt.plot(time3, U_N31, label='Numerical Solution Fun2 (2^10)')
plt.plot(time_true, U_T3, label='Exact Solution', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('U(t)')
plt.title('Numerical–Exact Solution Comparison (Fun2)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('3.png', dpi=300)  # 保存图像

plt.figure(4, figsize=(8, 6))  
plt.plot(time1, error12 + 1e-16, label='Error 1 (2^4)')
plt.plot(time2, error22, label='Error 2 (2^8)')
plt.plot(time3, error32, label='Error 3 (2^10)')

plt.xlabel('Time (s)')
plt.ylabel('Absolute Error')
plt.title('Absolute Error (Fun2)')
plt.legend()
plt.grid(True, which='both')  # 网格显示，both 使得主次网格都显示
plt.yscale('log')  # 关键：y轴对数刻度
plt.tight_layout()
plt.savefig('4.png', dpi=300)  # 保存图像

# 分别绘图
plt.figure(5, figsize=(8, 6))  
plt.plot(time1, U_N11, label='Numerical Solution Fun3 (2^4)',marker='o', markersize=5)
plt.plot(time2, U_N21, label='Numerical Solution Fun3 (2^8)')
plt.plot(time3, U_N31, label='Numerical Solution Fun3 (2^10)')
plt.plot(time_true, U_T3, label='Exact Solution', linestyle='--')

plt.xlabel('Time (s)')
plt.ylabel('U(t)')
plt.title('Numerical–Exact Solution Comparison (Fun3)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('5.png', dpi=300)  # 保存图像

plt.figure(6, figsize=(8, 6))  
plt.plot(time1, error12 + 1e-16, label='Error 1 (2^4)')
plt.plot(time2, error22, label='Error 2 (2^8)')
plt.plot(time3, error32, label='Error 3 (2^10)')

plt.xlabel('Time (s)')
plt.ylabel('Absolute Error')
plt.title('Absolute Error (Fun3)')
plt.legend()
plt.grid(True, which='both')  # 网格显示，both 使得主次网格都显示
plt.yscale('log')  # 关键：y轴对数刻度
plt.tight_layout()
plt.savefig('6.png', dpi=300)  # 保存图像

# 三个方法的比较
methods = ['Euler', 'Integral', 'Taylor-2nd']
markers = {'Euler': 'o', 'Integral': 's', 'Taylor-2nd': '^'}
linestyles = {'Euler': '-', 'Integral': '--', 'Taylor-2nd': '-.'}
colors = {'Euler': '#1f77b4', 'Integral': '#2ca02c', 'Taylor-2nd': '#d62728'}

Funi_list = [1, 2, 3]
hi_list = [1, 2, 3]
h_vals = [get_time_step(hi) for hi in hi_list]

# 结果存储
max_errors = {method: [] for method in methods}
l2_errors = {method: [] for method in methods}
run_times  = {method: [] for method in methods}

# 精确解
U_T = {hi: UTrue_function(hi) for hi in hi_list}

# 主循环
for Funi, method in zip(Funi_list, methods):
    for hi in hi_list:
        h = get_time_step(hi)
        t0 = time.time()
        U_num = example_function(hi, Funi, u_0)
        elapsed = time.time() - t0
        U_exact = U_T[hi]
        err = np.abs(np.array(U_num) - np.array(U_exact))
        max_errors[method].append(np.max(err))
        l2_errors[method].append(np.sqrt(np.mean(err**2)))
        run_times[method].append(elapsed)

# --------- 绘图 ---------
plt.figure(7, figsize=(8, 6))  
for method in methods:
    y = np.array(max_errors[method])
    plt.plot(h_vals, y,
             marker=markers[method],
             linestyle=linestyles[method],
             zorder=3,
             label=f'{method}')
    plt.fill_between(h_vals, y * 0.5, y * 1.5, alpha=0.2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Step Size $h$ (log scale)')
plt.ylabel('Max Absolute Error (log scale)')
plt.title('Max Error vs Step Size')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.savefig('error_vs_h.png', dpi=300)

plt.figure(8, figsize=(8, 6))  
for method in methods:
    y = np.clip(np.array(run_times[method]), 1e-6, None)
    color = colors[method]
    plt.plot(h_vals, y,
             marker=markers[method],
             linestyle=linestyles[method],
             zorder=3,
             label=f'{method}')
    plt.fill_between(h_vals,
                     y * 0.5, y * 1.5,
                     alpha=0.2,
                     zorder=2)

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Step Size $h$ (log scale)')
plt.ylabel('Runtime (seconds, log scale)')
plt.title('Computation Time vs Step Size')
plt.grid(True, which='both')
plt.legend()
plt.tight_layout()
plt.savefig('time_vs_h.png', dpi=300)
plt.show()