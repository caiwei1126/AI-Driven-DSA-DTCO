import matplotlib.pyplot as plt
import numpy as np

f = 0.28
Rg = 1
wall = 0.2 * Rg
grid_size = 0.25
L0 = 4.625 * Rg
diameter_top = 1.5 * L0
length_top = 3.5 * L0
diameter_bottom = 1.0 * L0
length_bottom = 2.5 * L0
diameter_box = diameter_top + 2 * wall
length_box = length_top + 2 * wall
height = 2 * L0
x_points = int(np.round(length_box / grid_size))
y_points = int(np.round(diameter_box / grid_size))
z_points = int(np.round(height / grid_size))
x = np.linspace(-length_box / 2, length_box / 2, x_points)
y = np.linspace(-diameter_box / 2, diameter_box / 2, y_points)
z = np.linspace(0, height, z_points)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
a = 1.5
b = 0.0
c = 1.5
mask = np.zeros_like(X, dtype=bool)
Lambda_r = np.zeros_like(X)
bottom_condition = np.zeros_like(mask, dtype=bool)
side_wall_condition = np.zeros_like(mask, dtype=bool)
z_ratio = z / height
diameter_z_all = diameter_bottom + (diameter_top - diameter_bottom) * z_ratio
length_z_all = length_bottom + (length_top - length_bottom) * z_ratio
radius_z_all = diameter_z_all / 2
long_z_all = length_z_all - diameter_z_all
for i in range(z_points):
    diameter = diameter_z_all[i]
    length = length_z_all[i]
    radius = radius_z_all[i]
    long = long_z_all[i]
    X_slice = X[:, :, i]
    Y_slice = Y[:, :, i]
    mask_rectangle = (
        (X_slice >= -long / 2) & (X_slice <= long / 2)
        & (Y_slice >= -radius) & (Y_slice <= radius)
    )
    mask_left_circle = ((X_slice + long / 2) ** 2 + Y_slice ** 2 <= radius ** 2) & (X_slice <= -long / 2)
    mask_right_circle = ((X_slice - long / 2) ** 2 + Y_slice ** 2 <= radius ** 2) & (X_slice >= long / 2)
    mask_base = mask_rectangle | mask_left_circle | mask_right_circle
    gaussian_bottom = a * np.exp(-(X_slice - b) ** 2 / (2 * c ** 2)) - radius
    gaussian_top = -a * np.exp(-(X_slice - b) ** 2 / (2 * c ** 2)) + radius
    mask_top = Y_slice <= gaussian_top
    mask_bottom = Y_slice >= gaussian_bottom
    mask_slice = mask_base & mask_top & mask_bottom
    mask[:, :, i] = mask_slice
    if i == 0:
        bottom_condition[:, :, i] = mask_slice
    delta_y = 0.35
    gaussian_top_inner = gaussian_top - delta_y
    gaussian_bottom_inner = gaussian_bottom + delta_y
    top_edge = (Y_slice <= gaussian_top) & (Y_slice >= gaussian_top_inner)
    bottom_edge = (Y_slice >= gaussian_bottom) & (Y_slice <= gaussian_bottom_inner)
    mask_x_range = (X_slice >= -long / 2 - 0.2) & (X_slice <= long / 2 + 0.2)
    side_wall_top = mask_base & top_edge & mask_x_range
    side_wall_bottom = mask_base & bottom_edge & mask_x_range
    distance_left = np.sqrt((X_slice + long / 2) ** 2 + Y_slice ** 2)
    distance_right = np.sqrt((X_slice - long / 2) ** 2 + Y_slice ** 2)
    arc_thickness = radius / x_points * 6.5
    wall_left = mask_left_circle & (np.abs(distance_left - radius) <= arc_thickness)
    wall_right = mask_right_circle & (np.abs(distance_right - radius) <= arc_thickness)
    side_wall_condition[:, :, i] = wall_left | wall_right | side_wall_top | side_wall_bottom
phi_s_wall = 0.1
phi_s_bottom = 0.1
Lambda_r[bottom_condition] = 50
Lambda_r[side_wall_condition] = 50

plt.figure(figsize=(6, 5))
plt.title("Lambda_r at Top (Z = height)")
plt.contourf(X[:, :, -1], Y[:, :, -1], Lambda_r[:, :, -1], levels=20, cmap="seismic")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.colorbar(label="Lambda_r")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.title("Lambda_r at Bottom (Z = 0)")
plt.contourf(X[:, :, 0], Y[:, :, 0], Lambda_r[:, :, 0], levels=20, cmap="seismic")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.colorbar(label="Lambda_r")
plt.tight_layout()
plt.show()

y_index_center = np.argmin(np.abs(y))

plt.figure(figsize=(6, 5))
plt.title("Lambda_r at Y ≈ 0 (Z-X plane)")
plt.contourf(X[:, y_index_center, :], Z[:, y_index_center, :], Lambda_r[:, y_index_center, :], levels=20, cmap="seismic")
plt.xlabel("X")
plt.ylabel("Z")
plt.axis("equal")
plt.colorbar(label="Lambda_r")
plt.tight_layout()
plt.show()
