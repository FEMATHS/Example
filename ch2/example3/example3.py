import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from sympy.utilities.lambdify import lambdify
from scipy.integrate import solve_ivp

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


# 定义符号
t = sp.symbols('t')
u = sp.Function('u')(t)

# 定义 f = u' = t - u^2
f = t - u**2

def true_solution(t0, hi, T, u0):
    def rhs(t, u): return t - u**2
    sol = solve_ivp(rhs, [t0, T], [u0], t_eval=np.linspace(t0, T, get_time_step(hi)))
    return sol.t, sol.y[0]

def get_time_step(hi):
    if hi == 1:
        return 2 ** 4
    elif hi == 2:
        return 2 ** 8
    elif hi == 3:
        return 2 ** 10
    else:
        raise ValueError("hi must be 1, 2, or 3")

# 自动求高阶导数,可以实现求导order阶导
def higher_order_derivatives(order=1):
    derivatives = [f]
    for i in range(1, order):
        d = sp.diff(derivatives[-1], t)
        # 替换所有 u^{(n)} 为之前导出的表达式
        for j in range(1, i + 1):
            d = d.subs(sp.Derivative(u, (t, j)), derivatives[j - 1])
        derivatives.append(d)
    return derivatives


def Taylor_Func(order=1):
  # 替换 sympy 表达式中的 u(t) 为符号 u
  derivatives_numeric = [d.subs(u, u_sym) for d in higher_order_derivatives(order)]
  # 转换为可调用的函数
  funcs = [lambdify((t, u_sym), d) for d in derivatives_numeric]
  return funcs

def Taylor_solution(t0,T,hi,u0,funcs):
  h = (T - t0) / get_time_step(hi)
  t, u = t0, u0
  t_vals = [t0]
  u_vals = [u0]
  # 使用高阶导数泰勒展开
  for _ in range(get_time_step(hi)):
      u_next = u
      for i, fi in enumerate(funcs):
          term = h**(i+1) / np.math.factorial(i+1) * fi(t, u)
          u_next += term
      u = u_next
      t += h
      t_vals.append(t)
      u_vals.append(u)
  return t_vals,u_vals

# 创建 u 变量，用于数值传入
u_sym = sp.symbols('u')
subs_dict = {u: u_sym}
# 设置初始条件
t0 = 0
u0 = 0
T = 1

order1 = 2 # 设置阶数
order2 = 5 # 设置阶数
order3 = 8 # 设置阶数
t1,u1=Taylor_solution(t0,T,1,u0,Taylor_Func(order1))
t2,u2=Taylor_solution(t0,T,1,u0,Taylor_Func(order2))
t3,u3=Taylor_solution(t0,T,1,u0,Taylor_Func(order3))
t4,u4=Taylor_solution(t0,T,2,u0,Taylor_Func(order3))
t5,u5=Taylor_solution(t0,T,3,u0,Taylor_Func(order3))
t_ref, u_ref = true_solution(t0, 1, T, u0)
t_ref1, u_ref1 = true_solution(t0, 2, T, u0)
t_ref2, u_ref2 = true_solution(t0, 3, T, u0)
error1 = np.abs(np.array(u1) - np.interp(np.array(t1), t_ref, u_ref))
error2 = np.abs(np.array(u2) - np.interp(np.array(t2), t_ref, u_ref))
error3 = np.abs(np.array(u3) - np.interp(np.array(t3), t_ref, u_ref))
error4 = np.abs(np.array(u4) - np.interp(np.array(t4), t_ref1, u_ref1))
error5 = np.abs(np.array(u5) - np.interp(np.array(t5), t_ref2, u_ref2))


# 绘图
plt.figure(1, figsize=(8, 6))
plt.plot(t1, u1, label="2-order Taylor", linestyle='--', color='blue', marker='o')
plt.plot(t2, u2, label="5-order Taylor", linestyle='-', color='green', marker='o')
plt.plot(t3, u3, label="8-order Taylor", linestyle=':', color='red', marker='o')

plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Taylor Series Methods stride 2^4 (2 to 8 Order)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('1.png', dpi=300)

plt.figure(2, figsize=(8, 6))
plt.plot(t1, error1, label="2-order Taylor", linestyle='--', color='blue', marker='o')
plt.plot(t2, error2, label="5-order Taylor", linestyle='-', color='green', marker='o')
plt.plot(t3, error3, label="8-order Taylor", linestyle=':', color='red', marker='o')

plt.xlabel("t")
plt.ylabel("Abs error")
plt.title("Taylor Series Error stride 2^4 (2 to 8 Order)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('2.png', dpi=300)

plt.figure(3, figsize=(8, 6))
plt.plot(t3, u3, label="stride 2^4", linestyle='--', color='blue', marker='o')
plt.plot(t4, u4, label="stride 2^8", linestyle='-', color='green')
plt.plot(t5, u5, label="stride 2^10", linestyle=':', color='red')

plt.xlabel("t")
plt.ylabel("u(t)")
plt.title("Taylor Series Methods 8 Order (2^4 to 2^10 stride)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('3.png', dpi=300)

plt.figure(4, figsize=(8, 6))
plt.plot(t3, np.array(error3) + 1e-16, label="stride 2^4", linestyle='--', color='blue', marker='o')
plt.plot(t4, error4, label="stride 2^8", linestyle='-', color='green')
plt.plot(t5, error5, label="stride 2^10", linestyle=':', color='red')

plt.xlabel("t")
plt.ylabel("u(t)")
plt.yscale('log')  
plt.title("Taylor Series Error 8 Order (2^4 to 2^10 stride)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('4.png', dpi=300)
plt.show()