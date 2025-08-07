import numpy as np
import matplotlib.pyplot as plt

# 图像全局字体设置
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 24
plt.rcParams['axes.titlesize'] = 24
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20
plt.rcParams['figure.titlesize'] = 24

def get_time_step(hi):
    if hi == 1:
        return 1 / 64
    elif hi == 2:
        return 1 / 128
    elif hi == 3:
        return 1 / 256
    else:
        raise ValueError("hi must be 1 , 2 or 3")

def solve_bvp(h, example_id=1):
    if example_id == 1:
        a, b = 0, 1
        N = int((b - a) / h)
        x = np.linspace(a, b, N + 1)
        u = np.linspace(1, 0.5, N + 1)
        u[0] = 1.0
        u[-1] = 0.5

        def f_rhs(xi, ui):
            return ((1 - xi) * ui + 1) / ((1 + xi) ** 2)

        def df_du(xi):
            return (1 - xi) / ((1 + xi) ** 2)

    elif example_id == 2:
        a, b = 0, np.pi
        N = int((b - a) / h)
        x = np.linspace(a, b, N + 1)
        u = np.linspace(-2, np.exp(np.pi) + 3, N + 1)
        u[0] = -2.0
        u[-1] = np.exp(np.pi) + 3

    elif example_id == 3:
        a, b = 0, np.pi
        N = int((b - a) / h)
        x = np.linspace(a, b, N + 1)
        u = x + 2 * np.sin(x)   # 更贴近真实解，有助于迭代收敛
        u[0] = 0.0
        u[-1] = np.pi

    else:
        raise ValueError("Invalid example_id")

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

            if example_id == 1:
                F[i - 1] = (uim1 - 2 * ui + uip1) / h**2 - f_rhs(xi, ui)
                if i > 1:
                    J[i - 1, i - 2] = 1 / h**2
                J[i - 1, i - 1] = -2 / h**2 - df_du(xi)
                if i < N - 1:
                    J[i - 1, i] = 1 / h**2

            elif example_id == 2:
                F[i - 1] = (uim1 - 2 * ui + uip1) / h**2 - (uip1 - uim1) / (2 * h) + ui - (np.exp(xi) - 3 * np.sin(xi))
                if i > 1:
                    J[i - 1, i - 2] = 1 / h**2 + 1 / (2 * h)
                J[i - 1, i - 1] = -2 / h**2 + 1
                if i < N - 1:
                    J[i - 1, i] = 1 / h**2 - 1 / (2 * h)

            elif example_id == 3:
                F[i - 1] = (
                    (uim1 - 2 * ui + uip1) / h**2
                    - xi * (uip1 - uim1) / (2 * h)
                    + ui
                    - (-2 * xi * np.cos(xi) + xi)  
                )
                if i > 1:
                    J[i - 1, i - 2] = 1 / h**2 + xi / (2 * h)
                J[i - 1, i - 1] = -2 / h**2 + 1
                if i < N - 1:
                    J[i - 1, i] = 1 / h**2 - xi / (2 * h)

        delta_u = np.linalg.solve(J, -F)
        u[1:N] += delta_u
        if np.linalg.norm(delta_u, ord=np.inf) < tol:
            break

    return x, u

def exact_solution(x, example_id=1):
    if example_id == 1:
        return 1 / (1 + x)
    elif example_id == 2:
        return np.exp(x) - 3 * np.cos(x)
    elif example_id == 3:
        return x + 2 * np.sin(x)
    else:
        raise ValueError("Invalid example_id")

def plot_results_for_example(example_id):
    plt.figure(1, figsize=(8, 6))  
    for hi in [1, 2, 3]:
        h = get_time_step(hi)
        x, u_num = solve_bvp(h, example_id=example_id)
        u_exact = exact_solution(x, example_id=example_id)
        plt.plot(x, u_num, label=f'Step (h=1/{int(1/h)})')

    plt.plot(x, u_exact, '--', label='Exact')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'FDM Numerical vs Exact Solution (Example {example_id})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'example{example_id}_1.png', dpi=300)
    plt.close()

    plt.figure(2, figsize=(8, 6)) 
    for hi in [1, 2, 3]:
        h = get_time_step(hi)
        x, u_num = solve_bvp(h, example_id=example_id)
        u_exact = exact_solution(x, example_id=example_id)
        error = np.abs(u_num - u_exact)
        print(f"Example {example_id}, h={h:.5f}, max error={np.max(error):.2e}")
        plt.semilogy(x, error + 1e-16, label=f'Step (h=1/{int(1/h)})')

    plt.xlabel('x')
    plt.ylabel('Abs Error (log scale)')
    plt.title(f'FDM Method Max Error vs Step Size (Example {example_id})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'example{example_id}_2.png', dpi=300)
    plt.close()

# ========== 主程序绘图 ==========

# 批量运行所有算例
for example_id in [1, 2, 3]:
    plot_results_for_example(example_id)

