import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import warnings
warnings.filterwarnings('ignore')

# Set font properties
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['font.size'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 18

class DiffusionSolver:
    def __init__(self, h=1/16, r=0.5, layers=8):
        self.h = h
        self.r = r
        self.tau = r * h * h
        self.N = int(1/h)
        self.layers = layers
        
    def initialize(self):
        """Initialize with strong heat source at center"""
        u = np.zeros(self.N + 1)
        # Create a stronger, wider initial heat source
        center = self.N // 2
        width = max(3, self.N // 16)  # Wider heat source
        
        for i in range(center - width, center + width + 1):
            if 0 <= i <= self.N:
                # Gaussian-like distribution for smoother initial condition
                distance = abs(i - center)
                u[i] = 1.0 * np.exp(-(distance / width) ** 2)
        
        return u
    
    def thomas_algorithm(self, a, b, c, d):
        """Thomas algorithm for tridiagonal system"""
        n = len(d)
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        x = np.zeros(n)
        
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n):
            m = b[i] - a[i] * c_prime[i-1]
            c_prime[i] = c[i] / m
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / m
        
        x[-1] = d_prime[-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
        return x
    
    def solve_explicit(self):
        """Classical explicit scheme"""
        u = self.initialize()
        history = [u.copy()]
        
        for j in range(self.layers):
            u_new = np.zeros(self.N + 1)
            for i in range(1, self.N):
                u_new[i] = self.r * u[i+1] + (1 - 2*self.r) * u[i] + self.r * u[i-1]
            u = u_new
            history.append(u.copy())
        
        return np.array(history)
    
    def solve_implicit(self):
        """Classical implicit scheme"""
        u = self.initialize()
        history = [u.copy()]
        
        for j in range(self.layers):
            n = self.N - 1
            a = np.full(n, -self.r)
            b = np.full(n, 1 + 2*self.r)
            c = np.full(n, -self.r)
            d = u[1:self.N]
            
            x = self.thomas_algorithm(a, b, c, d)
            u_new = np.zeros(self.N + 1)
            u_new[1:self.N] = x
            u = u_new
            history.append(u.copy())
        
        return np.array(history)
    
    def solve_crank_nicolson(self):
        """Crank-Nicolson (six-point symmetric scheme)"""
        u = self.initialize()
        history = [u.copy()]
        
        for j in range(self.layers):
            n = self.N - 1
            a = np.full(n, -self.r/2)
            b = np.full(n, 1 + self.r)
            c = np.full(n, -self.r/2)
            
            d = np.zeros(n)
            for i in range(n):
                d[i] = self.r/2 * u[i+2] + (1-self.r) * u[i+1] + self.r/2 * u[i]
            
            x = self.thomas_algorithm(a, b, c, d)
            u_new = np.zeros(self.N + 1)
            u_new[1:self.N] = x
            u = u_new
            history.append(u.copy())
        
        return np.array(history)


def create_padded_data(data):
    """Pad data with zeros to fill the entire time-space domain"""
    layers, spatial_points = data.shape
    # Create full matrix filled with zeros (cold/blue region)
    full_data = np.zeros((layers, spatial_points))
    
    # Fill in the computed values
    for t in range(layers):
        full_data[t, :] = data[t, :]
    
    return full_data


def create_long_heatmap_animation(method='explicit', save_path='diffusion_long_heatmap.gif'):
    """Create long-form heatmap animation (realistic thermal imaging style)"""
    
    # More layers for longer simulation
    solver = DiffusionSolver(h=1/64, r=0.5, layers=150)
    
    print(f"Solving {method} scheme...")
    if method == 'explicit':
        data = solver.solve_explicit()
        title = 'Classical Explicit Scheme - Heat Diffusion'
        color_label = 'red'
    elif method == 'implicit':
        data = solver.solve_implicit()
        title = 'Classical Implicit Scheme - Heat Diffusion'
        color_label = 'blue'
    else:  # crank_nicolson
        data = solver.solve_crank_nicolson()
        title = 'Crank-Nicolson Scheme - Heat Diffusion'
        color_label = 'green'
    
    x = np.linspace(0, 1, solver.N + 1)
    
    # Create full padded data matrix
    full_data = np.zeros((len(data), len(x)))
    
    # Create long-form layout (width >> height)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), 
                                     gridspec_kw={'height_ratios': [4, 1]})
    fig.suptitle(title, fontsize=24, fontweight='bold', y=0.98)
    
    # Use bwr colormap with fixed range [0, 1]
    cmap = 'bwr'
    vmin, vmax = 0, 1.0
    
    # Colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, 
                        norm=plt.Normalize(vmin=vmin, vmax=vmax)), 
                        ax=ax1, orientation='vertical', pad=0.02)
    cbar.set_label('Temperature u(x,t)', fontsize=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    
    # Information text box
    info_box = ax2.text(0.02, 0.95, '', transform=ax2.transAxes,
                       fontsize=16, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85),
                       family='monospace')
    
    plt.tight_layout()
    
    def init():
        """Initialize animation"""
        ax1.clear()
        ax2.clear()
        
        # Initialize with full cold background
        full_data[0, :] = data[0, :]
        
        im = ax1.imshow(full_data.T, aspect='auto', cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation='bilinear',
                        extent=[0, 1, 0, 1], origin='lower')
        ax1.set_xlabel('Normalized Time (0 to 1)', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Position x', fontsize=20, fontweight='bold')
        ax1.set_title('Heat Propagation: Red=Hot, Blue=Cold', fontsize=20, pad=15)
        
        line, = ax2.plot(x, data[0], color=color_label, linewidth=3, marker='o', 
                         markersize=4, markevery=4)
        ax2.fill_between(x, 0, data[0], alpha=0.3, color=color_label)
        ax2.set_xlabel('Position x', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Temperature u', fontsize=20, fontweight='bold')
        ax2.set_title('Current Temperature Profile', fontsize=20, pad=15)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(vmin, vmax * 1.1)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        
        return []
    
    def update(frame):
        """Update animation"""
        ax1.clear()
        ax2.clear()
        
        # Update full_data: copy current computed values
        for t in range(frame + 1):
            full_data[t, :] = data[t, :]
        
        # Calculate normalized time position (0 to 1)
        time_fraction = frame / (len(data) - 1)
        
        # Update heatmap - show full domain [0,1] x [0,1]
        im = ax1.imshow(full_data.T, aspect='auto', cmap=cmap,
                        vmin=vmin, vmax=vmax, interpolation='bilinear',
                        extent=[0, 1, 0, 1], origin='lower')
        ax1.set_xlabel('Normalized Time (0 to 1)', fontsize=20, fontweight='bold')
        ax1.set_ylabel('Position x', fontsize=20, fontweight='bold')
        ax1.set_title('Heat Propagation: Red=Hot, Blue=Cold', fontsize=20, pad=15)
        ax1.set_xlim(0, 1)
        
        # Draw vertical line to show current time
        ax1.axvline(x=time_fraction, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
        
        # Update curve
        line, = ax2.plot(x, data[frame], color=color_label, linewidth=3, marker='o', 
                         markersize=4, markevery=4)
        ax2.fill_between(x, 0, data[frame], alpha=0.3, color=color_label)
        ax2.set_xlabel('Position x', fontsize=20, fontweight='bold')
        ax2.set_ylabel('Temperature u', fontsize=20, fontweight='bold')
        ax2.set_title('Current Temperature Profile', fontsize=20, pad=15)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(vmin, vmax * 1.1)
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=1.5)
        
        # Update information
        max_idx = np.argmax(data[frame])
        info = f'Layer: {frame:3d}/{len(data)-1} | Real Time: {frame * solver.tau:.6f} | Progress: {time_fraction*100:.1f}%\n'
        info += f'Max Temp: {data[frame].max():.4f} @ x={x[max_idx]:.4f} | Min Temp: {data[frame].min():.4f}\n'
        info += f'Average Temp: {np.mean(data[frame]):.4f} | Energy: {np.sum(data[frame]) * solver.h:.4f}'
        info_box.set_text(info)
        
        return []
    
    print(f"Generating long-form heatmap animation...")
    anim = FuncAnimation(fig, update, init_func=init, frames=len(data),
                        interval=80, repeat=True)
    
    print(f"Saving animation to {save_path}...")
    writer = PillowWriter(fps=12)
    anim.save(save_path, writer=writer, dpi=120)
    
    print(f"✓ Long-form heatmap animation saved to: {save_path}")
    plt.close()


def create_ultra_wide_heatmap(method='explicit', save_path='diffusion_ultra_wide.gif'):
    """Create ultra-wide heatmap (realistic thermal imaging view)"""
    
    solver = DiffusionSolver(h=1/80, r=0.5, layers=180)
    
    print(f"Solving {method} scheme...")
    if method == 'explicit':
        data = solver.solve_explicit()
        title = 'Classical Explicit Scheme - Ultra-Wide View'
    elif method == 'implicit':
        data = solver.solve_implicit()
        title = 'Classical Implicit Scheme - Ultra-Wide View'
    else:
        data = solver.solve_crank_nicolson()
        title = 'Crank-Nicolson Scheme - Ultra-Wide View'
    
    x = np.linspace(0, 1, solver.N + 1)
    full_data = np.zeros((len(data), len(x)))
    
    # Ultra-wide ratio: 24:5
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111)
    
    cmap = 'bwr'
    vmin, vmax = 0, 1.0
    
    # Colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, 
                        norm=plt.Normalize(vmin=vmin, vmax=vmax)), 
                        ax=ax, orientation='horizontal', 
                        pad=0.12, aspect=50, shrink=0.8)
    cbar.set_label('Temperature (°C)', fontsize=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    
    # Time marker
    time_text = ax.text(0.01, 0.97, '', transform=ax.transAxes,
                       fontsize=18, color='yellow', fontweight='bold',
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))
    
    plt.tight_layout()
    
    def init():
        ax.clear()
        full_data[0, :] = data[0, :]
        
        im = ax.imshow(full_data.T, aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation='gaussian',
                       extent=[0, 1, 0, 1], origin='lower')
        ax.set_xlabel('Normalized Time (0 to 1)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Position (m)', fontsize=20, fontweight='bold')
        ax.set_title(title + ' | 1D Heat Conduction: Watch Heat Spread!', 
                    fontsize=22, fontweight='bold', pad=20)
        ax.set_xlim(0, 1)
        ax.grid(True, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        
        for y_pos in [0, 0.25, 0.5, 0.75, 1.0]:
            ax.axhline(y=y_pos, color='gray', linestyle=':', linewidth=1, alpha=0.4)
        
        return []
    
    def update(frame):
        ax.clear()
        
        for t in range(frame + 1):
            full_data[t, :] = data[t, :]
        
        time_fraction = frame / (len(data) - 1)
        
        im = ax.imshow(full_data.T, aspect='auto', cmap=cmap,
                       vmin=vmin, vmax=vmax, interpolation='gaussian',
                       extent=[0, 1, 0, 1], origin='lower')
        
        ax.set_xlabel('Normalized Time (0 to 1)', fontsize=20, fontweight='bold')
        ax.set_ylabel('Position (m)', fontsize=20, fontweight='bold')
        ax.set_title(title + ' | 1D Heat Conduction: Watch Heat Spread!', 
                    fontsize=22, fontweight='bold', pad=20)
        ax.set_xlim(0, 1)
        ax.grid(True, color='gray', linestyle=':', linewidth=1, alpha=0.3)
        
        for y_pos in [0, 0.25, 0.5, 0.75, 1.0]:
            ax.axhline(y=y_pos, color='gray', linestyle=':', linewidth=1, alpha=0.4)
        
        # Current time line
        ax.axvline(x=time_fraction, color='yellow', linestyle='--', linewidth=2, alpha=0.8)
        
        time_text.set_text(f'Time: {frame * solver.tau:.5f}s | Layer {frame}/{len(data)-1}\nProgress: {time_fraction*100:.1f}%')
        
        return []
    
    print(f"Generating ultra-wide heatmap animation...")
    anim = FuncAnimation(fig, update, init_func=init, frames=len(data),
                        interval=60, repeat=True)
    
    print(f"Saving animation to {save_path}...")
    writer = PillowWriter(fps=15)
    anim.save(save_path, writer=writer, dpi=100)
    
    print(f"✓ Ultra-wide heatmap animation saved to: {save_path}")
    plt.close()


def create_combined_comparison(save_path='diffusion_comparison.gif'):
    """Create side-by-side comparison of all three methods"""
    
    solver = DiffusionSolver(h=1/64, r=0.5, layers=120)
    
    print("Solving all three schemes...")
    explicit = solver.solve_explicit()
    implicit = solver.solve_implicit()
    cn = solver.solve_crank_nicolson()
    
    x = np.linspace(0, 1, solver.N + 1)
    
    # Create full data matrices for all three methods
    full_explicit = np.zeros((len(explicit), len(x)))
    full_implicit = np.zeros((len(implicit), len(x)))
    full_cn = np.zeros((len(cn), len(x)))
    
    # Create three-column layout
    fig, axes = plt.subplots(1, 3, figsize=(28, 8))
    fig.suptitle('Heat Diffusion Comparison: Red=Hot spreading through Blue=Cold', 
                 fontsize=24, fontweight='bold', y=0.98)
    
    # Setup for each subplot
    methods_data = [
        (explicit, full_explicit, 'Classical Explicit', 'red'),
        (implicit, full_implicit, 'Classical Implicit', 'blue'),
        (cn, full_cn, 'Crank-Nicolson', 'green')
    ]
    
    vmin, vmax = 0, 1.0
    cmap = 'bwr'
    
    # Shared colorbar
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, 
                        norm=plt.Normalize(vmin=vmin, vmax=vmax)), 
                        ax=axes, orientation='horizontal',
                        pad=0.08, aspect=40, shrink=0.8)
    cbar.set_label('Temperature: Blue=Cold(0) → Red=Hot(1)', fontsize=20, fontweight='bold')
    cbar.ax.tick_params(labelsize=16)
    
    # Global time indicator
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=18,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    def init():
        for idx, (data, full_data, title, color) in enumerate(methods_data):
            full_data[0, :] = data[0, :]
            ax = axes[idx]
            ax.clear()
            im = ax.imshow(full_data.T, aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax, interpolation='bilinear',
                          extent=[0, 1, 0, 1], origin='lower')
            ax.set_xlabel('Normalized Time', fontsize=18, fontweight='bold')
            ax.set_ylabel('Position x', fontsize=18, fontweight='bold')
            ax.set_title(title, fontsize=20, fontweight='bold', color=color, pad=15)
            ax.set_xlim(0, 1)
        return []
    
    def update(frame):
        time_fraction = frame / (len(explicit) - 1)
        
        for idx, (data, full_data, title, color) in enumerate(methods_data):
            for t in range(frame + 1):
                full_data[t, :] = data[t, :]
            
            ax = axes[idx]
            ax.clear()
            im = ax.imshow(full_data.T, aspect='auto', cmap=cmap,
                          vmin=vmin, vmax=vmax, interpolation='bilinear',
                          extent=[0, 1, 0, 1], origin='lower')
            ax.set_xlabel('Normalized Time', fontsize=18, fontweight='bold')
            ax.set_ylabel('Position x', fontsize=18, fontweight='bold')
            ax.set_title(title, fontsize=20, fontweight='bold', color=color, pad=15)
            ax.set_xlim(0, 1)
            ax.axvline(x=time_fraction, color='yellow', linestyle='--', linewidth=2, alpha=0.7)
        
        max_vals = [explicit[frame].max(), implicit[frame].max(), cn[frame].max()]
        time_text.set_text(
            f'Layer {frame}/{len(explicit)-1} | Time t = {frame * solver.tau:.6f} | Progress: {time_fraction*100:.1f}% | '
            f'Peak Temps: {max_vals[0]:.3f}, {max_vals[1]:.3f}, {max_vals[2]:.3f}'
        )
        
        return []
    
    print(f"Generating comparison animation...")
    anim = FuncAnimation(fig, update, init_func=init, frames=len(explicit),
                        interval=80, repeat=True)
    
    print(f"Saving animation to {save_path}...")
    writer = PillowWriter(fps=12)
    anim.save(save_path, writer=writer, dpi=120)
    
    print(f"✓ Comparison animation saved to: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 80)
    print("HEAT DIFFUSION VISUALIZATION - Watch Red Heat Spread Through Blue Cold!")
    print("=" * 80)
    
    methods = ['explicit', 'implicit', 'crank_nicolson']
    method_names = ['Classical Explicit', 'Classical Implicit', 'Crank-Nicolson']
    
    # 1. Generate long-form heatmaps
    print("\n[PART 1: Long-Form Heatmap Animations]")
    print("-" * 80)
    for i, (method, name) in enumerate(zip(methods, method_names)):
        print(f"\n[{i+1}/3] Generating {name} long-form heatmap...")
        create_long_heatmap_animation(
            method=method,
            save_path=f'diffusion_long_{method}.gif'
        )
    
    # 2. Generate ultra-wide heatmaps
    print("\n" + "=" * 80)
    print("[PART 2: Ultra-Wide View Heatmaps]")
    print("-" * 80)
    for i, (method, name) in enumerate(zip(methods, method_names)):
        print(f"\n[{i+1}/3] Generating {name} ultra-wide view...")
        create_ultra_wide_heatmap(
            method=method,
            save_path=f'diffusion_ultra_{method}.gif'
        )
    
    # 3. Generate comparison view
    print("\n" + "=" * 80)
    print("[PART 3: Three-Method Comparison View]")
    print("-" * 80)
    print("\nGenerating side-by-side comparison...")
    create_combined_comparison(save_path='diffusion_comparison.gif')
    
    print("\n" + "=" * 80)
    print("✓ ALL ANIMATIONS GENERATED SUCCESSFULLY!")
    print("\nGenerated Files:")
    print("\n[Long-Form Heatmaps] - 150 layers, strong heat source")
    print("  • diffusion_long_explicit.gif")
    print("  • diffusion_long_implicit.gif")
    print("  • diffusion_long_crank_nicolson.gif")
    print("\n[Ultra-Wide View Heatmaps] - 180 layers, strong heat source")
    print("  • diffusion_ultra_explicit.gif")
    print("  • diffusion_ultra_implicit.gif")
    print("  • diffusion_ultra_crank_nicolson.gif")
    print("\n[Comparison View] - 120 layers, strong heat source")
    print("  • diffusion_comparison.gif")
    print("\nVisualization: Blue=Cold(0°C) → Red=Hot(1°C)")
    print("Watch the red heat gradually spread through the blue cold material!")
    print("=" * 80)