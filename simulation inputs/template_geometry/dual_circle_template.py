import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

f = 0.28
Rg = 1
wall = 0.2 * Rg
L0 = 4.2 * Rg
cd = 1.939 * L0
diameter_box = cd + 2 * wall
ccd = 1.224 * L0
length_box = 3.163 * L0 + 2 * wall
height = 2 * L0
x_points = 70
y_points = 45
z_points = 46
x = np.linspace(-length_box / 2, length_box / 2, x_points)
y = np.linspace(-diameter_box / 2, diameter_box / 2, y_points)
z = np.linspace(0, height, z_points)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
radius = cd / 2
center1_x = -ccd / 2
center2_x = ccd / 2
mask_left_circle = ((X - center1_x) ** 2 + Y ** 2 <= radius ** 2)
mask_right_circle = ((X - center2_x) ** 2 + Y ** 2 <= radius ** 2)
mask = mask_left_circle | mask_right_circle
phi_s_wall = 0.1
phi_s_bottom = 0.1
Lambda_r = np.zeros_like(X)
bottom_condition = mask & (Z == 0)
Lambda_r[bottom_condition] = 50
distance_to_left_circle = np.sqrt((X - center1_x) ** 2 + Y ** 2)
distance_to_right_circle = np.sqrt((X - center2_x) ** 2 + Y ** 2)
side_wall_left_circle = (mask_left_circle & (np.abs(distance_to_left_circle - radius) <= (radius / x_points * 4)) & ~mask_right_circle)
side_wall_right_circle = (mask_right_circle & (np.abs(distance_to_right_circle - radius) <= (radius / x_points * 4)) & ~mask_left_circle)
side_wall_condition_x0 = (
    (np.abs(distance_to_left_circle - radius) <= (radius / x_points * 4))
    | (np.abs(distance_to_right_circle - radius) <= (radius / x_points * 4))
) & (X == 0)
side_wall_condition = side_wall_left_circle | side_wall_right_circle | side_wall_condition_x0
Lambda_r[side_wall_condition] = 50

bottom_plane_index = np.isclose(z, 0, atol=1e-5).nonzero()[0][0]
Lambda_r_bottom = Lambda_r[:, :, bottom_plane_index]
plt.figure(figsize=(8, 6))
plt.imshow(
    Lambda_r_bottom.T,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="viridis",
    aspect="equal",
)
plt.colorbar(label="Lambda_r")
plt.title("bottom (Z=0)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()

middle_plane_index = np.isclose(z, height / 2, atol=0.2).nonzero()[0][0]
Lambda_r_middle = Lambda_r[:, :, middle_plane_index]
plt.figure(figsize=(8, 6))
plt.imshow(
    Lambda_r_middle.T,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="viridis",
    aspect="equal",
)
plt.colorbar(label="Lambda_r")
plt.title("Middle Plane (Z=height/2)")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()

y_zero_plane_index = np.isclose(y, 0, atol=0.2).nonzero()[0][0]
Lambda_r_y_zero = Lambda_r[:, y_zero_plane_index, :]
plt.figure(figsize=(8, 6))
plt.imshow(
    Lambda_r_y_zero.T,
    extent=[x.min(), x.max(), z.min(), z.max()],
    origin="lower",
    cmap="viridis",
    aspect="auto",
)
plt.colorbar(label="Lambda_r")
plt.title("Z-X Plane (Y=0)")
plt.xlabel("X")
plt.ylabel("Z")
plt.grid()
plt.show()
