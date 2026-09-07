import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# 2D pollution propagation
#
# g_t + u g_x + w g_z
# = mu g_xx + eta g_zz - sigma g + f
# ============================================================

# -----------------------------
# Domain and grid
# -----------------------------
Lx = 20.0
H = 6.0

Nx = 240
Nz = 100

dx = Lx / (Nx - 1)
dz = H / (Nz - 1)

x = np.linspace(0, Lx, Nx)
z = np.linspace(0, H, Nz)

X, Z = np.meshgrid(x, z, indexing="ij")

# -----------------------------
# Parameters
# -----------------------------
mu = 0.012
eta = 0.006
sigma = 0.015
a = 0.25

T = 35.0


# -----------------------------
# Time-dependent wind field
# -----------------------------
def velocity(t):
    u = (
        1.0
        + 0.45 * Z / H
        + 0.35 * np.sin(2 * np.pi * t / 12.0)
        + 0.20 * np.sin(2 * np.pi * Z / H)
    )

    w = (
        0.15
        * np.sin(np.pi * X / Lx)
        * np.sin(np.pi * Z / H)
        * np.cos(2 * np.pi * t / 10.0)
    )

    return u, w


# -----------------------------
# Time step
# -----------------------------
max_u_est = 2.2
max_w_est = 0.2

dt_adv_x = dx / max_u_est
dt_adv_z = dz / max_w_est
dt_diff_x = (dx**2) / (2 * mu)
dt_diff_z = (dz**2) / (2 * eta)
# dt_diff = 0.25 / (mu / dx**2 + eta / dz**2)

dt = 0.35 * min(dt_adv_x, dt_adv_z, dt_diff_x, dt_diff_z)
Nt = int(T / dt)


# -----------------------------
# Initial condition
# -----------------------------
g = np.zeros((Nx, Nz))

g += 1.0 * np.exp(-((X - 3.0) ** 2 / 0.50 + (Z - 1.0) ** 2 / 0.20))
g += 0.8 * np.exp(-((X - 6.0) ** 2 / 0.70 + (Z - 2.5) ** 2 / 0.35))
g += 0.5 * np.exp(-((X - 11.0) ** 2 / 1.20 + (Z - 1.8) ** 2 / 0.45))


# -----------------------------
# Source term
# -----------------------------
def gaussian_source(xs, zs, sx, sz):
    return np.exp(-((X - xs) ** 2 / sx + (Z - zs) ** 2 / sz))


def source(t):
    f1 = (
        1.5
        * (1.0 + 0.7 * np.sin(2 * np.pi * t / 8.0))
        * gaussian_source(2.0, 0.45, 0.35, 0.05)
    )

    f2 = (
        1.2 * np.exp(-((t - 10.0) ** 2) / 20.0) * gaussian_source(7.0, 0.55, 0.50, 0.06)
    )

    f3 = 3.0 * np.exp(-((t - 18.0) ** 2) / 1.5) * gaussian_source(4.5, 1.4, 0.35, 0.18)

    xs = 10.0 + 2.0 * np.sin(2 * np.pi * t / 15.0)

    f4 = (
        0.8
        * np.exp(-((t - 24.0) ** 2) / 40.0)
        * np.exp(-((X - xs) ** 2 / 0.60 + (Z - 3.0) ** 2 / 0.35))
    )

    return f1 + f2 + f3 + f4


# -----------------------------
# Boundary conditions
# -----------------------------
def apply_boundary_conditions(g):
    # Side boundaries: g = 0
    g[0, :] = 0.0
    # g[-1, :] = 0.0

    # Bottom boundary: dg/dz = a g
    g[:, 0] = g[:, 1] / (1.0 + a * dz)

    # Top boundary: dg/dz = 0
    g[:, -1] = g[:, -2]

    return g


# -----------------------------
# One explicit Euler step
# -----------------------------
def step(g, t):
    g = apply_boundary_conditions(g)

    gn = g.copy()

    u, w = velocity(t)

    G = gn[1:-1, 1:-1]
    U = u[1:-1, 1:-1]
    W = w[1:-1, 1:-1]

    # Upwind derivative in x
    gx_backward = (gn[1:-1, 1:-1] - gn[:-2, 1:-1]) / dx
    gx_forward = (gn[2:, 1:-1] - gn[1:-1, 1:-1]) / dx
    Dx_up = np.where(U >= 0.0, gx_backward, gx_forward)

    # Upwind derivative in z
    gz_backward = (gn[1:-1, 1:-1] - gn[1:-1, :-2]) / dz
    gz_forward = (gn[1:-1, 2:] - gn[1:-1, 1:-1]) / dz
    Dz_up = np.where(W >= 0.0, gz_backward, gz_forward)

    # Central diffusion
    Dxx = (gn[2:, 1:-1] - 2.0 * G + gn[:-2, 1:-1]) / dx**2
    Dzz = (gn[1:-1, 2:] - 2.0 * G + gn[1:-1, :-2]) / dz**2

    F = source(t)[1:-1, 1:-1]

    g[1:-1, 1:-1] = (
        G - dt * (U * Dx_up + W * Dz_up) + dt * (mu * Dxx + eta * Dzz - sigma * G + F)
    )

    g[g < 0.0] = 0.0

    return apply_boundary_conditions(g)


# -----------------------------
# Time integration
# -----------------------------
frames = []
times = []

save_every = 8

for n in range(Nt):
    t = n * dt
    g = step(g, t)

    if n % save_every == 0:
        frames.append(g.copy())
        times.append(t)

print(f"Stored {len(frames)} frames.")

# -----------------------------
# Animation
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

# Concentration plot
vmax_g = np.percentile(np.array(frames), 99.5)

im = ax1.imshow(
    frames[0].T,
    origin="lower",
    extent=[0, Lx, 0, H],
    aspect="auto",
    vmin=0,
    vmax=vmax_g,
    cmap="viridis",
)

cbar1 = plt.colorbar(im, ax=ax1)
cbar1.set_label("pollutant concentration")

ax1.set_xlabel("horizontal distance x")
ax1.set_ylabel("height z")
title1 = ax1.set_title(f"Pollution, t = {times[0]:.2f}")

# Wind plot
skip = (slice(None, None, 12), slice(None, None, 6))

u0, w0 = velocity(times[0])
speed0 = np.sqrt(u0**2 + w0**2)

quiv = ax2.quiver(
    X[skip],
    Z[skip],
    u0[skip],
    w0[skip],
    speed0[skip],
    cmap="plasma",
    scale=25,
)

cbar2 = plt.colorbar(quiv, ax=ax2)
cbar2.set_label("wind speed")

ax2.set_xlim(0, Lx)
ax2.set_ylim(0, H)
ax2.set_xlabel("horizontal distance x")
ax2.set_ylabel("height z")
title2 = ax2.set_title("Wind direction and magnitude")


def update(k):
    t = times[k]

    im.set_data(frames[k].T)
    title1.set_text(f"Pollution, t = {t:.2f}")

    u, w = velocity(t)
    speed = np.sqrt(u**2 + w**2)

    quiv.set_UVC(u[skip], w[skip], speed[skip])
    title2.set_text(f"Wind, t = {t:.2f}")

    return im, quiv, title1, title2


ani = FuncAnimation(
    fig,
    update,
    frames=len(frames),
    interval=60,
    blit=False,
)

plt.tight_layout()
plt.show()

# Optional: save as GIF
# ani.save("pollution_2d_with_wind.gif", writer="pillow", fps=20)
