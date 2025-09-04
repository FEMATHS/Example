import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# 图像全局字体设置
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 20

# ========== 1. 定义正六边形区域 ==========
R = 1.0  # 外接圆半径
verts = [(R * math.cos(k * math.pi / 3.0), R * math.sin(k * math.pi / 3.0)) for k in range(6)]

def point_in_hexagon(x, y, vertices, tol=1e-12):
    """判断 (x,y) 是否在凸多边形（六边形）内"""
    sign = 0
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        ex, ey = (x2 - x1, y2 - y1)
        vx, vy = (x - x1, y - y1)
        cross = ex * vy - ey * vx
        if abs(cross) <= tol:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

# ========== 2. 生成网格 ==========
h = 1.0 / 3.0   # 网格步长
xs = np.arange(-R, R + h/2, h)
ys = np.arange(-R, R + h/2, h)
nx, ny = len(xs), len(ys)

inside = np.zeros((ny, nx), dtype=bool)
for j, y in enumerate(ys):
    for i, x in enumerate(xs):
        if point_in_hexagon(x, y, verts):
            inside[j, i] = True

def neighbors(i, j):
    return [(i+1, j), (i-1, j), (i, j+1), (i, j-1)]

# 找边界点
boundary = np.zeros_like(inside, dtype=bool)
for j in range(ny):
    for i in range(nx):
        if not inside[j, i]:
            continue
        for (ii, jj) in neighbors(i, j):
            if ii < 0 or ii >= nx or jj < 0 or jj >= ny or not inside[jj, ii]:
                boundary[j, i] = True
                break
interior = inside & (~boundary)

# ========== 3. 构造稀疏线性方程组 ==========
index_map = -np.ones((ny, nx), dtype=int)
coords = []
k = 0
for j in range(ny):
    for i in range(nx):
        if interior[j, i]:
            index_map[j, i] = k
            coords.append((i, j))
            k += 1

N = len(coords)
A = lil_matrix((N, N))
b = np.full((N,), h*h)  # f=1

for row, (i, j) in enumerate(coords):
    A[row, row] = 4.0
    for (ii, jj) in neighbors(i, j):
        if ii < 0 or ii >= nx or jj < 0 or jj >= ny or not inside[jj, ii]:
            continue
        if interior[jj, ii]:
            col = index_map[jj, ii]
            A[row, col] = -1.0
        # 邻居是边界点 → u=0，不贡献未知量

# ========== 4. 求解 ==========
u_int = spsolve(A.tocsr(), b) if N > 0 else np.array([])

# ========== 5. 恢复到全场 ==========
U = np.zeros((ny, nx))
for j in range(ny):
    for i in range(nx):
        if interior[j, i]:
            U[j, i] = u_int[index_map[j, i]]
        else:
            U[j, i] = 0.0

# ========== 6. 可视化 ==========
plt.figure(figsize=(6, 5))
plt.imshow(U, extent=[xs[0], xs[-1], ys[0], ys[-1]], origin="lower", cmap="coolwarm")
plt.colorbar(label="u(x,y)")
plt.title("Poisson equation solution in hexagon (h=1/3)")
plt.gca().set_aspect("equal")
plt.savefig("solution_hexagon.png", dpi=300, bbox_inches="tight")  # 保存解图像

# ========== 7. 网格点可视化 ==========
X, Y = np.meshgrid(xs, ys)

plt.figure(figsize=(6,6))

# 画出全局网格线（灰色）
for xi in xs:
    plt.plot([xi, xi], [ys[0], ys[-1]], color="lightgray", linewidth=0.8)
for yi in ys:
    plt.plot([xs[0], xs[-1]], [yi, yi], color="lightgray", linewidth=0.8)

# 内部点（蓝色）
plt.scatter(X[interior], Y[interior], c="blue", s=20, label="Interior")

# 边界点（红色）
plt.scatter(X[boundary], Y[boundary], c="red", s=30, label="Boundary")
for idx, (i, j) in enumerate(coords):
    plt.text(xs[i], ys[j], str(idx), color='black', fontsize=25, ha='center', va='center')

plt.gca().set_aspect("equal")
plt.legend()
plt.title("Grid points in hexagon (h=1/8) with mesh lines")
plt.savefig("solution_hexagon.png", dpi=300, bbox_inches="tight")  # 保存解图像
plt.show()

