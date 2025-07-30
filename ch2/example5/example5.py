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

def AM_method(hi=1, u_0=1, order=4):
    assert order in [1, 2, 3, 4, 5], "只支持 AM1 到 AM5"
    h = get_time_step(hi)
    N = int(1 / h)

    # AM 方法的组合系数（已按表格缩放过）
    coeff_dict = {
        1: [1],
        2: [1/2, 1/2],
        3: [5/12, 8/12, -1/12],
        4: [9/24, 19/24, -5/24, 1/24],
        5: [251/720, 646/720, -264/720, 106/720, -19/720]
    }
    coeffs = coeff_dict[order]

    # 初始用 RK4
    U = Runge_Kutta4(hi=hi, Funi=1, u_0=u_0)
    f_list = [f(i*h, U[i]) for i in range(order)]

    for n in range(order, N + 1):
        # 预测值（用 AB 方法）
        f_predict = sum(c * f_list[-(i + 1)] for i, c in enumerate(coeffs[1:]))
        u_predict = U[n - 1] + h * f_predict  # 初始预测 u_m（忽略 f_m）

        # 迭代求解隐式项 u_m（Picard 法）
        for _ in range(3):  # 通常 2~3 次即可收敛
            f0 = f(n * h, u_predict)
            u_predict = U[n - 1] + h * (coeffs[0] * f0 + f_predict)

        U[n] = u_predict
        f_list.append(f(n * h, U[n]))
        if len(f_list) > order:
            f_list.pop(0)

    return U


def Gear_method(hi=1, u_0=1, order=4):
    assert order in [1, 2, 3, 4, 5, 6], "只支持 Gear1 到 Gear6"
    h = get_time_step(hi)
    N = int(1 / h)

    # Gear 系数表（c_{k,i}, g_k）
    coeff_table = {
        1: ([], 1),
        2: ([-4/3, 1/3], 2/3),
        3: ([-18/11, 9/11, -2/11], 6/11),
        4: ([-48/25, 36/25, -16/25, 3/25], 12/25),
        5: ([-300/137, 300/137, -200/137, 75/137, -12/137], 60/137),
        6: ([-360/147, 450/147, -400/147, 225/147, -72/147, 10/147], 60/147)
    }
    coeffs, gk = coeff_table[order]

    # 初始值
    U = Runge_Kutta4(hi=hi, Funi=1, u_0=u_0)

    for n in range(order, N + 1):
        # 处理 order=1 的特殊情况
        if order == 1:
            rhs = 0
        else:
            rhs = -sum(coeffs[i] * U[n - i - 1] for i in range(order))
        
        # 初值预测
        u_predict = U[n - 1]

        # Picard 迭代
        for _ in range(3):
            f_val = f(n * h, u_predict)
            u_predict = h * gk * f_val + rhs

        U[n] = u_predict

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

# 添加AM方法比较图
plt.figure(4, figsize=(8, 6))
plt.plot(t_vals, U_exact, label='Exact', linestyle='--', color='black')

for k in range(1, 6):
    U_am = np.array(AM_method(hi=1, u_0=u_0, order=k))
    plt.plot(t_vals, U_am + 1e-16, label=f'AM{k}', marker='s', markersize=5)

plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Comparison of Adams–Moulton (Order 1–5)")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('4.png', dpi=300)  # 保存图像

# AM方法误差图
plt.figure(5, figsize=(8, 6))

for k in range(1, 6):
    U_am = np.array(AM_method(hi=1, u_0=u_0, order=k))
    error = np.abs(U_am - U_exact)
    plt.semilogy(t_vals, error + 1e-16, label=f'Error (AM{k})', marker='s', markersize=5)

plt.xlabel("t")
plt.ylabel("Absolute Error (log scale)")
plt.title("Error of Adams–Moulton (Order 1–5)")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('5.png', dpi=300)  # 保存图像

# 添加Gear方法比较图
plt.figure(6, figsize=(8, 6))
plt.plot(t_vals, U_exact, label='Exact', linestyle='--', color='black')

for k in range(1, 7):
    U_gear = np.array(Gear_method(hi=1, u_0=u_0, order=k))
    plt.plot(t_vals, U_gear + 1e-16, label=f'Gear{k}', marker='^', markersize=5)

plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Comparison of Gear Methods (Order 1–6)")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('6.png', dpi=300)  # 保存图像

# Gear方法误差图
plt.figure(7, figsize=(8, 6))

for k in range(1, 7):
    U_gear = np.array(Gear_method(hi=1, u_0=u_0, order=k))
    error = np.abs(U_gear - U_exact)
    plt.semilogy(t_vals, error + 1e-16, label=f'Error (Gear{k})', marker='^', markersize=5)

plt.xlabel("t")
plt.ylabel("Absolute Error (log scale)")
plt.title("Error of Gear Methods (Order 1–6)")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('7.png', dpi=300)  # 保存图像

# 三种方法综合比较图
plt.figure(8, figsize=(8, 6))
plt.plot(t_vals, U_exact, label='Exact', linestyle='--', color='black', linewidth=2)

# 选择4阶方法进行比较
U_ab4 = np.array(AB_method(hi=1, u_0=u_0, order=4))
U_am4 = np.array(AM_method(hi=1, u_0=u_0, order=4))
U_gear4 = np.array(Gear_method(hi=1, u_0=u_0, order=4))

plt.plot(t_vals, U_ab4 + 1e-16, label='AB4', marker='o', markersize=6)
plt.plot(t_vals, U_am4 + 1e-16, label='AM4', marker='s', markersize=6)
plt.plot(t_vals, U_gear4 + 1e-16, label='Gear4', marker='^', markersize=6)

plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Comparison of 4th Order Methods: AB4 vs AM4 vs Gear4")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('8.png', dpi=300)  # 保存图像

# 三种方法误差综合比较
plt.figure(9, figsize=(8, 6))

error_ab4 = np.abs(U_ab4 - U_exact)
error_am4 = np.abs(U_am4 - U_exact)
error_gear4 = np.abs(U_gear4 - U_exact)

plt.semilogy(t_vals, error_ab4 + 1e-16, label='Error (AB4)', marker='o', markersize=6)
plt.semilogy(t_vals, error_am4 + 1e-16, label='Error (AM4)', marker='s', markersize=6)
plt.semilogy(t_vals, error_gear4 + 1e-16, label='Error (Gear4)', marker='^', markersize=6)

plt.xlabel("t")
plt.ylabel("Absolute Error (log scale)")
plt.title("Error Comparison: AB4 vs AM4 vs Gear4")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('9.png', dpi=300)  # 保存图像

# AM方法收敛性分析
plt.figure(10, figsize=(8, 6))

# 计算不同步长下的最大误差
h_values_am = []
max_errors_am = {1: [], 2: [], 3: [], 4: [], 5: []}

for hi in [1, 2, 3, 4]:
    h = get_time_step(hi)
    h_values_am.append(h)
    U_exact_hi = np.array(UTrue_function(hi=hi))
    
    for order in range(1, 6):
        U_am = np.array(AM_method(hi=hi, u_0=u_0, order=order))
        max_error = np.max(np.abs(U_am - U_exact_hi))
        max_errors_am[order].append(max_error)

# 绘制AM方法收敛性曲线
for order in range(1, 6):
    plt.loglog(h_values_am, max_errors_am[order], 's-', label=f'AM{order}')

plt.xlabel("Step size h")
plt.ylabel("Maximum Error (log scale)")
plt.title("Convergence Analysis of Adams–Moulton Methods")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('10.png', dpi=300)  # 保存图像

# Gear方法收敛性分析
plt.figure(11, figsize=(8, 6))

# 计算不同步长下的最大误差
h_values_gear = []
max_errors_gear = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

for hi in [1, 2, 3, 4]:
    h = get_time_step(hi)
    h_values_gear.append(h)
    U_exact_hi = np.array(UTrue_function(hi=hi))
    
    for order in range(1, 7):
        U_gear = np.array(Gear_method(hi=hi, u_0=u_0, order=order))
        max_error = np.max(np.abs(U_gear - U_exact_hi))
        max_errors_gear[order].append(max_error)

# 绘制Gear方法收敛性曲线
for order in range(1, 7):
    plt.loglog(h_values_gear, max_errors_gear[order], '^-', label=f'Gear{order}')

plt.xlabel("Step size h")
plt.ylabel("Maximum Error (log scale)")
plt.title("Convergence Analysis of Gear Methods")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('11.png', dpi=300)  # 保存图像

# 三种方法4阶收敛性综合比较
plt.figure(12, figsize=(8, 6))

# 重新计算AB方法的收敛性（为了保持一致性）
h_values_ab = []
max_errors_ab4 = []

for hi in [1, 2, 3, 4]:
    h = get_time_step(hi)
    h_values_ab.append(h)
    U_exact_hi = np.array(UTrue_function(hi=hi))
    U_ab4 = np.array(AB_method(hi=hi, u_0=u_0, order=4))
    max_error = np.max(np.abs(U_ab4 - U_exact_hi))
    max_errors_ab4.append(max_error)

# 绘制三种4阶方法的收敛性比较
plt.loglog(h_values_ab, max_errors_ab4, 'o-', label='AB4', linewidth=2, markersize=8)
plt.loglog(h_values_am, max_errors_am[4], 's-', label='AM4', linewidth=2, markersize=8)
plt.loglog(h_values_gear, max_errors_gear[4], '^-', label='Gear4', linewidth=2, markersize=8)

plt.xlabel("Step size h")
plt.ylabel("Maximum Error (log scale)")
plt.title("Convergence Comparison: AB4 vs AM4 vs Gear4")
plt.legend()
plt.grid(True, linestyle='--')
plt.tight_layout()
plt.savefig('12.png', dpi=300)  # 保存图像

plt.show()