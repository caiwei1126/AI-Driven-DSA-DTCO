import matplotlib.pyplot as plt
import numpy as np

f = 0.28
Rg = 1
wall = 0.2 * Rg
grid_size = 0.25
L0 = 4.2 * Rg
cd_top = 2.296 * L0
CDY_top = 3.582 * L0
cd_bottom = 1.296 * L0
CDY_bottom = 2.582 * L0
diameter_box = cd_top + 2 * wall
length_box = CDY_top + 2 * wall
height = 2 * L0
x_points = int(np.round(length_box / grid_size))
y_points = int(np.round(diameter_box / grid_size))
z_points = int(np.round(height / grid_size))
x = np.linspace(-length_box / 2, length_box / 2, x_points)
y = np.linspace(-diameter_box / 2, diameter_box / 2, y_points)
z = np.linspace(0, height, z_points)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
Lambda_r = np.zeros_like(X)
mask = np.zeros_like(X, dtype=bool)
bottom_condition = np.zeros_like(mask, dtype=bool)
side_wall_condition = np.zeros_like(mask, dtype=bool)
z_ratio = z / height
cd_z_all = cd_bottom + (cd_top - cd_bottom) * z_ratio
CDY_z_all = CDY_bottom + (CDY_top - CDY_bottom) * z_ratio
ccd_z_all = CDY_z_all - cd_z_all
radius_all = cd_z_all / 2
center1_x_all = -ccd_z_all / 2
center2_x_all = ccd_z_all / 2
wall_thickness_all = radius_all / x_points * 5
for i in range(z_points):
    r = radius_all[i]
    c1 = center1_x_all[i]
    c2 = center2_x_all[i]
    wt = wall_thickness_all[i]
    X_slice = X[:, :, i]
    Y_slice = Y[:, :, i]
    mask_left_circle = ((X_slice - c1) ** 2 + Y_slice ** 2 <= r ** 2)
    mask_right_circle = ((X_slice - c2) ** 2 + Y_slice ** 2 <= r ** 2)
    mask_slice = mask_left_circle | mask_right_circle
    mask[:, :, i] = mask_slice
    if i == 0:
        bottom_condition[:, :, i] = mask_slice
    distance_to_left_circle = np.sqrt((X_slice - c1) ** 2 + Y_slice ** 2)
    distance_to_right_circle = np.sqrt((X_slice - c2) ** 2 + Y_slice ** 2)
    side_wall_left_circle = (
        mask_left_circle
        & (np.abs(distance_to_left_circle - r) <= wt)
        & ~mask_right_circle
    )
    side_wall_right_circle = (
        mask_right_circle
        & (np.abs(distance_to_right_circle - r) <= wt)
        & ~mask_left_circle
    )
    side_wall_condition_x0 = (
        (np.abs(distance_to_left_circle - r) <= wt)
        | (np.abs(distance_to_right_circle - r) <= wt)
    ) & (np.abs(X_slice) < grid_size / 2)
    side_wall_condition[:, :, i] = (
        side_wall_left_circle | side_wall_right_circle | side_wall_condition_x0
    )
phi_s_wall = 0.15
phi_s_bottom = 0.15
Lambda_r[bottom_condition] = 50
Lambda_r[side_wall_condition] = 50

plt.figure(figsize=(6, 5))
plt.title("Lambda_r at Bottom (Z = 0)")
plt.contourf(X[:, :, 0], Y[:, :, 0], Lambda_r[:, :, 0], levels=20, cmap="seismic")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis("equal")
plt.colorbar(label="Lambda_r")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
plt.title("Lambda_r at Top (Z = height)")
plt.contourf(X[:, :, -1], Y[:, :, -1], Lambda_r[:, :, -1], levels=20, cmap="seismic")
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
