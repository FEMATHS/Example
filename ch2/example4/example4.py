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

def Runge_Kutta2(hi=1, Funi=1, u_0=1):
    assert u_0 != 0, "初始值 u_0 不能为 0，否则会除以 0 崩溃"
    ti = get_time_step(hi)
    T = 1 / ti
    U = [0] * (int(T) + 1)
    U[0] = u_0

    for m in range(1, int(T) + 1):
        t_m = (m - 1) * ti
        u_m = U[m - 1]

        if Funi == 1:  # 中点法（修正 Euler 法）
            k1 = f(t_m, u_m)
            k2 = f(t_m + 0.5 * ti, u_m + 0.5 * ti * k1)
            U[m] = u_m + ti * k2

        elif Funi == 2:  # 标准二阶 Runge-Kutta 法（又称 Ralston 方法）
            k1 = f(t_m, u_m)
            k2 = f(t_m + ti, u_m + ti * k1)
            U[m] = u_m + 0.5 * ti * (k1 + k2)

        elif Funi == 3:  # Heun 二阶法
            k1 = f(t_m, u_m)
            k2 = f(t_m + (2/3) * ti, u_m + (2/3) * ti * k1)
            U[m] = u_m + (ti / 4) * (k1 + 3 * k2)

        else:
            raise ValueError("Funi must be 1 (Midpoint), 2 (RK2), or 3 (Heun)")
    
    return U

def Runge_Kutta3(hi=1, Funi=1, u_0=1):
    assert u_0 != 0, "初始值 u_0 不能为 0，否则会除以 0 崩溃"
    ti = get_time_step(hi)
    T = 1 / ti
    U = [0] * (int(T) + 1)
    U[0] = u_0

    for m in range(1, int(T) + 1):
        t_m = (m - 1) * ti
        u_m = U[m - 1]

        if Funi == 1:  # Kutta 三阶法
            k1 = f(t_m, u_m)
            k2 = f(t_m + 0.5 * ti, u_m + 0.5 * ti * k1)
            k3 = f(t_m + ti, u_m - ti * k1 + 2 * ti * k2)
            U[m] = u_m + (ti / 6) * (k1 + 4 * k2 + k3)

        elif Funi == 2:  # Heun 三阶法
            k1 = f(t_m, u_m)
            k2 = f(t_m + ti / 3, u_m + (ti / 3) * k1)
            k3 = f(t_m + 2 * ti / 3, u_m + (2 * ti / 3) * k2)
            U[m] = u_m + (ti / 4) * (k1 + 3 * k3)

        else:
            raise ValueError("Funi must be 1 (Kutta 3rd) or 2 (Heun 3rd)")

    return U


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


def UTrue_function(hi=1):
    ti = get_time_step(hi)
    T = 1 / ti
    U_T = [0] * (int(T) + 1)
    for m in range(0, int(T) + 1):
        U_T[m] = np.sqrt(1 + 2 * m * ti)
        # print("现在计算第", m * ti, "s的精确解为", U_T[m])
    return U_T

# 设置初始条件
u_0 = 1

# 数值解（hi=1，步长 h=2^-4）
U_N11 = Runge_Kutta2(hi=1, Funi=1, u_0=1)  # Midpoint 2rd
U_N12 = Runge_Kutta2(hi=1, Funi=2, u_0=1)  # RK2 2rd
U_N13 = Runge_Kutta2(hi=1, Funi=3, u_0=1)  # Heun 2rd

# 计算数值解
U_N21 = Runge_Kutta3(hi=1, Funi=1, u_0=1)  # Kutta 3rd
U_N22 = Runge_Kutta3(hi=1, Funi=2, u_0=1)  # Heun 3rd

U_N31 = Runge_Kutta4(hi=1, Funi=1, u_0=1)  # Classical RK4
U_N32 = Runge_Kutta4(hi=1, Funi=2, u_0=1)  # Kutta 4th
# 精确解与时间向量
U_T1 = UTrue_function(1)

time1 = np.linspace(0, 1.0, len(U_T1))

# 误差计算
error_mid = np.abs(np.array(U_N11) - np.array(U_T1))
error_rk2 = np.abs(np.array(U_N12) - np.array(U_T1))
error_heun = np.abs(np.array(U_N13) - np.array(U_T1))

error_3_kutta = np.abs(np.array(U_N21) - np.array(U_T1))
error_3_heun = np.abs(np.array(U_N22) - np.array(U_T1))

error_4_classic = np.abs(np.array(U_N31) - np.array(U_T1))
error_4_kutta = np.abs(np.array(U_N32) - np.array(U_T1))

U_N11_h3 = Runge_Kutta2(hi=3, Funi=1, u_0=1)  # Midpoint 2nd
U_N12_h3 = Runge_Kutta2(hi=3, Funi=2, u_0=1)  # RK2 2nd
U_N13_h3 = Runge_Kutta2(hi=3, Funi=3, u_0=1)  # Heun 2nd

U_N21_h3 = Runge_Kutta3(hi=3, Funi=1, u_0=1)  # Kutta 3rd
U_N22_h3 = Runge_Kutta3(hi=3, Funi=2, u_0=1)  # Heun 3rd

U_N31_h3 = Runge_Kutta4(hi=3, Funi=1, u_0=1)  # Classical RK4
U_N32_h3 = Runge_Kutta4(hi=3, Funi=2, u_0=1)  # Kutta 4th

# 精确解与时间向量
U_T1_h3 = UTrue_function(3)
time1_h3 = np.linspace(0, 1.0, len(U_T1_h3))

# 误差计算
error_mid_h3 = np.abs(np.array(U_N11_h3) - np.array(U_T1_h3))
error_rk2_h3 = np.abs(np.array(U_N12_h3) - np.array(U_T1_h3))
error_heun_h3 = np.abs(np.array(U_N13_h3) - np.array(U_T1_h3))

error_3_kutta_h3 = np.abs(np.array(U_N21_h3) - np.array(U_T1_h3))
error_3_heun_h3 = np.abs(np.array(U_N22_h3) - np.array(U_T1_h3))

error_4_classic_h3 = np.abs(np.array(U_N31_h3) - np.array(U_T1_h3))
error_4_kutta_h3 = np.abs(np.array(U_N32_h3) - np.array(U_T1_h3))

# ---------- 图像1：数值解 vs 精确解 ----------
plt.figure(1,figsize=(8, 6))
plt.title("2rd Order Runge-Kutta Methods ($h = 2^{-4}$)")

plt.plot(time1, U_T1, 'k-', label='True Solution')
plt.plot(time1, U_N11, 'b--', label='Midpoint')
plt.plot(time1, U_N12, 'g-.', label='RK2')
plt.plot(time1, U_N13, 'r:', label='Heun')

plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('1.png', dpi=300)

# ---------- 图像2：误差图 ----------
plt.figure(2,figsize=(8, 6))
plt.title("Error of RK 2nd Order Methods")

plt.plot(time1, error_mid, 'b--', label='Midpoint Error')
plt.plot(time1, error_rk2, 'g-.', label='RK2 Error')
plt.plot(time1, error_heun, 'r:', label='Heun Error')

plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('2.png', dpi=300)

# 三阶解图
plt.figure(3,figsize=(8, 6))
plt.title("3rd Order Runge-Kutta Methods ($h = 2^{-4}$)")
plt.plot(time1, U_T1, 'k-', label='True Solution')
plt.plot(time1, U_N21, 'b--', label='Kutta 3rd')
plt.plot(time1, U_N22, 'g-.', label='Heun 3rd')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('3.png', dpi=300)

# 三阶误差图
plt.figure(4,figsize=(8, 6))
plt.title("Error of RK 3nd Order Methods")
plt.plot(time1, error_3_kutta, 'b--', label='Kutta 3rd Error')
plt.plot(time1, error_3_heun, 'g-.', label='Heun 3rd Error')
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('4.png', dpi=300)

plt.figure(5,figsize=(8, 6))
plt.title("4th Order Runge-Kutta Methods ($h = 2^{-4}$)")
plt.plot(time1, U_T1, 'k-', label='True Solution')
plt.plot(time1, U_N31, 'r--', label='Classical RK4')
plt.plot(time1, U_N32, 'm-.', label='Kutta 4th')
plt.xlabel('t')
plt.ylabel('u(t)')
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('5.png', dpi=300)

# 四阶误差图
plt.figure(6,figsize=(8, 6))
plt.title("Error of 4th Order RK Methods")
plt.plot(time1, error_4_classic, 'r--', label='Classical RK4 Error')
plt.plot(time1, error_4_kutta, 'm-.', label='Kutta 4th Error')
plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.legend()
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('6.png', dpi=300)

plt.figure(7,figsize=(10, 7))
plt.title("Error Comparison of RK Methods ($h = 2^{-4}$)")

plt.plot(time1, error_mid, 'b--', label='Midpoint (2nd)')
plt.plot(time1, error_rk2, 'g-.', label='RK2 (2nd)')
plt.plot(time1, error_heun, 'r:', label='Heun (2nd)')

plt.plot(time1, error_3_kutta, 'c--', label='Kutta (3rd)')
plt.plot(time1, error_3_heun, 'm-.', label='Heun (3rd)')

plt.plot(time1, error_4_classic, 'y--', label='Classical RK4 (4th)')
plt.plot(time1, error_4_kutta, 'k-.', label='Kutta (4th)')

plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.yscale('log')
plt.legend(loc='upper left')
plt.grid(True)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('7.png', dpi=300)

plt.figure(8,figsize=(10, 7))
plt.title("Error Comparison of RK Methods ($h = 2^{-10}$)")

plt.plot(time1_h3, error_mid_h3, 'b--', label='Midpoint (2nd)')
plt.plot(time1_h3, error_rk2_h3, 'g-.', label='RK2 (2nd)')
plt.plot(time1_h3, error_heun_h3, 'r:', label='Heun (2nd)')

plt.plot(time1_h3, error_3_kutta_h3, 'c--', label='Kutta (3rd)')
plt.plot(time1_h3, error_3_heun_h3, 'm-.', label='Heun (3rd)')

plt.plot(time1_h3, error_4_classic_h3, 'y--', label='Classical RK4 (4th)')
plt.plot(time1_h3, error_4_kutta_h3, 'k-.', label='Kutta (4th)')

plt.xlabel('t')
plt.ylabel('Absolute Error')
plt.yscale('log')
plt.legend(loc='upper left')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('8.png', dpi=300)
plt.show()


