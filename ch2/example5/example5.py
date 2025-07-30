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
    elif hi == 4:
        return 1 / (2 ** 12)
    else:
        raise ValueError("hi must be 1, 2, or 3")

def f(t, u):
    return u - 2 * t / u

def UTrue_function(hi=1):
    ti = get_time_step(hi)
    T = 1 / ti
    U_T = [0] * (int(T) + 1)
    for m in range(0, int(T) + 1):
        U_T[m] = np.sqrt(1 + 2 * m * ti)
        # print("现在计算第", m * ti, "s的精确解为", U_T[m])
    return U_T

def Runge_Kutta4(hi=1, Funi=1, u_0=1):
    assert u_0 != 0, "初始值 u_0 不能为 0，否则会除以 0 崩溃"
    ti = get_time_step(hi)
    T = 1 / ti
    U = [0] * (int(T) + 1)
    U[0] = u_0

    for m in range(1, int(T) + 1):
        t_m = (m - 1) * ti
        u_m = U[m - 1]

        if Funi == 1:  # 古典四阶 Runge-Kutta 方法
            k1 = f(t_m, u_m)
            k2 = f(t_m + 0.5 * ti, u_m + 0.5 * ti * k1)
            k3 = f(t_m + 0.5 * ti, u_m + 0.5 * ti * k2)
            k4 = f(t_m + ti, u_m + ti * k3)
            U[m] = u_m + (ti / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        elif Funi == 2:  # Kutta 四阶方法
            k1 = f(t_m, u_m)
            k2 = f(t_m + ti / 3, u_m + (ti / 3) * k1)
            k3 = f(t_m + 2 * ti / 3, u_m - (ti / 3) * k1 + ti * k2)
            k4 = f(t_m + ti, u_m + ti * k1 - ti * k2 + ti * k3)
            U[m] = u_m + (ti / 8) * (k1 + 3 * k2 + 3 * k3 + k4)

        else:
            raise ValueError("Funi must be 1 (Classical RK4) or 2 (Kutta 4th)")

    return U

def AB_method(hi=1, u_0=1, order=4):
    assert order in [1, 2, 3, 4, 5], "只支持 AB1 到 AB5"
    h = get_time_step(hi)
    N = int(1 / h)

    # 系数表，按 [f_n, f_{n-1}, f_{n-2}, ...] 排列
    coeff_dict = {
        1: [1],
        2: [3/2, -1/2],
        3: [23/12, -16/12, 5/12],
        4: [55/24, -59/24, 37/24, -9/24],
        5: [1901/720, -2774/720, 2616/720, -1274/720, 251/720],
    }
    coeffs = coeff_dict[order]

    # 前几步用 Runge-Kutta 生成
    U = Runge_Kutta4(hi=hi, Funi=1, u_0=u_0)
    
    # 初始化 f 值列表
    f_list = []
    for i in range(order):
        f_list.append(f(i*h, U[i]))

    # 从 order 开始使用 AB 方法迭代
    for n in range(order, N + 1):
        # 计算新的 u 值
        f_terms = sum(c * f_list[-(i + 1)] for i, c in enumerate(coeffs))
        u_next = U[n - 1] + h * f_terms
        U[n] = u_next
        
        # 更新 f 值列表
        f_list.append(f(n * h, u_next))
        if len(f_list) > order:
            f_list.pop(0)  # 保持列表长度为 order

    return U

# 设置初始条件
u_0 = 1

# 数值解（hi=1，步长 h=2^-4）
U_exact = np.array(UTrue_function(hi=1))
t_vals = np.linspace(0, 1, len(U_exact))

plt.figure(1, figsize=(8, 6))
plt.plot(t_vals, U_exact, label='Exact', linestyle='--', color='black')

for k in range(1, 6):
    U_ab = np.array(AB_method(hi=1, u_0=u_0, order=k))
    plt.plot(t_vals, U_ab+ 1e-16, label=f'AB{k}',marker='o', markersize=5)

plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Comparison of Adams–Bashforth (Order 1–5)")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('1.png', dpi=300)  # 保存图像

plt.figure(2, figsize=(8, 6))

for k in range(1, 6):
    U_ab = np.array(AB_method(hi=1, u_0=u_0, order=k))
    error = np.abs(U_ab - U_exact)
    plt.semilogy(t_vals, error+ 1e-16, label=f'Error (AB{k})',marker='o', markersize=5)

plt.xlabel("t")
plt.ylabel("Absolute Error (log scale)")
plt.title("Error of Adams–Bashforth (Order 1–5)")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('2.png', dpi=300)  # 保存图像

# 添加收敛性分析图
plt.figure(3, figsize=(8, 6))

# 计算不同步长下的最大误差
h_values = []
max_errors = {1: [], 2: [], 3: [], 4: [], 5: []}

for hi in [1, 2, 3, 4]:
    h = get_time_step(hi)
    h_values.append(h)
    U_exact_hi = np.array(UTrue_function(hi=hi))
    
    for order in range(1, 6):
        U_ab = np.array(AB_method(hi=hi, u_0=u_0, order=order))
        max_error = np.max(np.abs(U_ab - U_exact_hi))
        max_errors[order].append(max_error)

# 绘制收敛性曲线
for order in range(1, 6):
    plt.loglog(h_values, max_errors[order], 'o-', label=f'AB{order}')

plt.xlabel("Step size h")
plt.ylabel("Maximum Error (log scale)")
plt.title("Convergence Analysis of Adams–Bashforth Methods")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('3.png', dpi=300)  # 保存图像
plt.show()