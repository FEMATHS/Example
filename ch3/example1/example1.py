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
import numpy as np
import matplotlib.pyplot as plt

def get_time_step(hi):
    if hi == 1:
        return 1 / 64
    elif hi == 2:
        return 1 / 128
    elif hi == 3:
        return 1 / 256
    else:
        raise ValueError("hi must be 1 , 2 or 3")

def f_rhs(xi, ui):
    return ((1 - xi) * ui + 1) / ((1 + xi) ** 2)

def solve_bvp(h):
    N = int(1 / h)
    x = np.linspace(0, 1, N + 1)
    
    # 初始 guess (可以使用解析解或者线性插值)
    u = np.linspace(1, 0.5, N + 1)
    
    # 边界条件
    u[0] = 1.0
    u[-1] = 0.5

    # 牛顿迭代求解非线性方程组
    max_iter = 100
    tol = 1e-10

    for _ in range(max_iter):
        F = np.zeros(N - 1)
        J = np.zeros((N - 1, N - 1))

        for i in range(1, N):
            xi = x[i]
            ui = u[i]
            uim1 = u[i - 1]
            uip1 = u[i + 1]

            F[i - 1] = (uim1 - 2 * ui + uip1) / h**2 - f_rhs(xi, ui)

            # 雅可比矩阵：三对角结构
            if i > 1:
                J[i - 1, i - 2] = 1 / h**2
            J[i - 1, i - 1] = -2 / h**2 - (1 - xi) / ((1 + xi)**2)
            if i < N - 1:
                J[i - 1, i] = 1 / h**2

        delta_u = np.linalg.solve(J, -F)
        u[1:N] += delta_u

        if np.linalg.norm(delta_u, ord=np.inf) < tol:
            break

    return x, u

def exact_solution(x):
    return 1 / (1 + x)

plt.figure(1, figsize=(8, 6))  
for hi in [1,2,3]:
    h = get_time_step(hi)
    x, u_num = solve_bvp(h)
    u_exact = exact_solution(x)
    plt.plot(x, u_num, label=f'Step (h=1/{int(1/h)})')

plt.plot(x, u_exact, '--', label='Exact')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.title('FDM Numerical vs Exact Solution')
plt.legend()
plt.grid(True)
plt.savefig('1.png', dpi=300)

plt.figure(2, figsize=(8, 6)) 

for hi in [1,2,3]:
    h = get_time_step(hi)
    x, u_num = solve_bvp(h)
    u_exact = exact_solution(x)
    error = np.abs(u_num - u_exact)
    print(f"h={h}, max error={np.max(error)}")  # Debug
    plt.semilogy(x, np.array(error)+ 1e-16, label=f'Step (h=1/{int(1/h)})')  # log scale

plt.xlabel('x')
plt.ylabel('Abs Error (log scale)')
plt.title('FDM Method Max Error vs Step Size')
plt.legend()
plt.grid(True)
plt.savefig('2.png', dpi=300)
plt.show()
