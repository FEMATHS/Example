import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, ticker

plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 20

# =========================
# 参数设置
# =========================
h = 1/16
r = 1/2
tau = r * h**2
x = np.arange(0, 1 + h, h)
Nx = len(x)
Nt = 8  # 时间层数

# 初值：中点扰动
u0 = np.zeros(Nx)
u0[Nx//2] = 1 / 2**10

# 三种方法初始值
u_exp = u0.copy()
u_imp = u0.copy()
u_cn  = u0.copy()

# 历史层存储
U_exp = [u_exp.copy()]
U_imp = [u_imp.copy()]
U_cn  = [u_cn.copy()]

# =========================
# 系数矩阵构造（隐式）
# =========================
A_imp = np.zeros((Nx-2, Nx-2))
A_cn  = np.zeros((Nx-2, Nx-2))
B_cn  = np.zeros((Nx-2, Nx-2))

for i in range(Nx-2):
    if i > 0:
        A_imp[i, i-1] = -r
        A_cn[i, i-1] = -r/2
        B_cn[i, i-1] =  r/2
    A_imp[i, i] = 1 + 2*r
    A_cn[i, i] = 1 + r
    B_cn[i, i] = 1 - r
    if i < Nx-3:
        A_imp[i, i+1] = -r
        A_cn[i, i+1] = -r/2
        B_cn[i, i+1] =  r/2

# =========================
# 时间推进
# =========================
for n in range(1, Nt+1):
    # 显式
    u_new = u_exp.copy()
    for i in range(1, Nx-1):
        u_new[i] = u_exp[i] + r * (u_exp[i-1] - 2*u_exp[i] + u_exp[i+1])
    u_exp = u_new.copy()
    U_exp.append(u_exp.copy())

    # 隐式
    b = u_imp[1:-1]
    u_inner = np.linalg.solve(A_imp, b)
    u_imp[1:-1] = u_inner
    U_imp.append(u_imp.copy())

    # Crank–Nicolson
    b_cn = B_cn @ u_cn[1:-1]
    u_inner = np.linalg.solve(A_cn, b_cn)
    u_cn[1:-1] = u_inner
    U_cn.append(u_cn.copy())

# 转为数组
U_exp = np.array(U_exp)
U_imp = np.array(U_imp)
U_cn  = np.array(U_cn)
T = np.linspace(0, Nt*tau, Nt+1)

# =========================
# 解析解 (近似正弦展开)
# =========================
def exact_solution(x, t):
    u = np.zeros_like(x)
    n_terms = 50
    for n in range(1, n_terms+1):
        lam = n * np.pi
        coeff = (2 / 1) * (1 / 2**10) * np.sin(lam/2) / lam
        u += coeff * np.sin(lam*x) * np.exp(-lam**2 * t)
    return u

U_exact = np.array([exact_solution(x, t) for t in T])

# =========================
# 绝对误差
# =========================
Err_exp = np.abs(U_exp - U_exact)
Err_imp = np.abs(U_imp - U_exact)
Err_cn  = np.abs(U_cn  - U_exact)

# =========================
# 自定义科学计数法显示函数 (LaTeX 风格，保留1位有效数字)
# =========================
def sci_formatter(x, pos):
    if x == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(x))))
    mantissa = x / 10**exponent
    return r"${:.1f} \times 10^{{{}}}$".format(mantissa, exponent)

formatter = ticker.FuncFormatter(sci_formatter)

# =========================
# 新版绘图函数
# =========================
def plot_surface(U, title, filename):
    X, TT = np.meshgrid(x, T)

    # 自动计算各轴和色条的数量级
    def get_exponent(arr):
        max_val = np.max(np.abs(arr))
        return int(np.floor(np.log10(max_val))) if max_val != 0 else 0

    exp_x = get_exponent(x)
    exp_t = get_exponent(T)
    exp_z = get_exponent(U)

    # 缩放数据
    X_scaled = X / 10**exp_x if exp_x != 0 else X
    TT_scaled = TT / 10**exp_t if exp_t != 0 else TT
    U_scaled = U / 10**exp_z if exp_z != 0 else U

    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(TT_scaled, X_scaled, U_scaled, cmap=cm.viridis)

    # 设置视角
    ax.view_init(elev=30, azim=60)

    # 坐标轴标签
    ax.set_xlabel(r"$t \, [\times 10^{{{}}}]$".format(exp_t))
    ax.set_ylabel(r"$x \, [\times 10^{{{}}}]$".format(exp_x))
    ax.set_zlabel(r"$u(x,t) \, [\times 10^{{{}}}]$".format(exp_z))

    # ---- 不显示标题 ----
    # ax.set_title(r"${}$".format(title))

    # colorbar 保留
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=8)
    cbar.set_label(r"$\text{{{}}} \, [\times 10^{{{}}}]$".format(title, exp_z))

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)





# =========================
# 生成三种方法及误差图
# =========================
plot_surface(U_exp, "Explicit Scheme (FTCS)", "Explicit_FTCS.png")
plot_surface(U_imp, "Implicit Scheme (BTCS)", "Implicit_BTCS.png")
plot_surface(U_cn,  "Crank–Nicolson Scheme", "Crank_Nicolson.png")

plot_surface(Err_exp, "Absolute Error (Explicit)", "Err_Explicit.png")
plot_surface(Err_imp, "Absolute Error (Implicit)", "Err_Implicit.png")
plot_surface(Err_cn,  "Absolute Error (Crank–Nicolson)", "Err_CN.png")

print("✅ 所有图片已生成，z轴和色条统一数量级显示，已保存本地。")
