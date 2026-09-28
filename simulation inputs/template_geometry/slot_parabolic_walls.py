import matplotlib.pyplot as plt
import numpy as np

f = 0.28
Rg = 1
wall = 0.25 * Rg
L0 = 4.8 * Rg
diameter = 1.5 * L0
diameter_box = diameter + 2 * wall
length = 3.5 * L0
long = length - diameter
length_box = length + 2 * wall
height = 2 * L0
grid_size = 0.25
x_points = int(np.round(length_box / grid_size))
y_points = int(np.round(diameter_box / grid_size))
z_points = int(np.round(height / grid_size))
x = np.linspace(-length_box / 2, length_box / 2, x_points)
y = np.linspace(-diameter_box / 2, diameter_box / 2, y_points)
z = np.linspace(0, height, z_points)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
radius = diameter / 2
mask_rectangle = (X >= -long / 2) & (X <= long / 2) & (Y >= -diameter / 2) & (Y <= diameter / 2)
mask_left_circle = ((X + long / 2) ** 2 + Y ** 2 <= radius ** 2) & (X <= -long / 2)
mask_right_circle = ((X - long / 2) ** 2 + Y ** 2 <= radius ** 2) & (X >= long / 2)
mask = mask_rectangle | mask_left_circle | mask_right_circle
a = 1.5
b = 0.0
c = 1.4
mask_bottom_gaussian = Y <= (a * np.exp(-(X - b) ** 2 / (2 * c ** 2)) - diameter / 2)
mask_top_gaussian = Y <= (-a * np.exp(-(X - b) ** 2 / (2 * c ** 2)) + diameter / 2)
mask = mask & (~mask_bottom_gaussian) & (mask_top_gaussian)
mask_bottom_gaussian_plus = Y <= (a * np.exp(-(X - b) ** 2 / (2 * c ** 2)) - diameter / 2 + 0.3)
mask_top_gaussian_minus = Y <= (-a * np.exp(-(X - b) ** 2 / (2 * c ** 2)) + diameter / 2 - 0.3)
mask_left_x = (-long / 2 - 0.2) <= X
mask_right_x = X <= (long / 2 + 0.2)
side_wall_bottom_gaussian = ~mask_bottom_gaussian & mask_bottom_gaussian_plus & mask_left_x & mask_right_x
side_wall_top_gaussian = mask_top_gaussian & (~mask_top_gaussian_minus) & mask_left_x & mask_right_x
phi_s_wall = 0.1
phi_s_bottom = 0.1
Lambda_r = np.zeros_like(X)
bottom_condition = mask & (Z == 0)
Lambda_r[bottom_condition] = 50
distance_to_left_circle = np.sqrt((X + long / 2) ** 2 + Y ** 2)
distance_to_right_circle = np.sqrt((X - long / 2) ** 2 + Y ** 2)
side_wall_left_circle = (mask_left_circle & (np.abs(distance_to_left_circle - radius) <= (radius / x_points * 5)))
side_wall_right_circle = (mask_right_circle & (np.abs(distance_to_right_circle - radius) <= (radius / x_points * 5)))
side_wall_condition = side_wall_left_circle | side_wall_right_circle | side_wall_top_gaussian | side_wall_bottom_gaussian
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

y_zero_plane_index = np.isclose(y, 0, atol=0.1).nonzero()[0][0]
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
