import numpy as np
import matplotlib.pyplot as plt

Rg = 1
wall = 0.05 * Rg
grid_size = 0.05

L0 = 4.8 * Rg
diameter = 1.2798 * L0
length = 3.106 * L0
neck_width = 0.8986 * L0
neck_width_corrected = neck_width - 0.2
height = 2 * L0
c = 1.35

diameter_box = diameter + 2 * wall
long = length - diameter
length_box = length + 2 * wall

x_points = int(np.round(length_box / grid_size))
y_points = int(np.round(diameter_box / grid_size))
z_points = int(np.round(height / grid_size))

x = np.linspace(-length_box / 2, length_box / 2, x_points)
y = np.linspace(-diameter_box / 2, diameter_box / 2, y_points)
z = np.linspace(0, height, z_points)

X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

radius = diameter / 2

mask_rectangle = (
    (X >= -long / 2) & (X <= long / 2) &
    (Y >= -diameter / 2) & (Y <= diameter / 2)
)
mask_left_circle = (
    ((X + long / 2) ** 2 + Y ** 2 <= radius ** 2) &
    (X <= -long / 2)
)
mask_right_circle = (
    ((X - long / 2) ** 2 + Y ** 2 <= radius ** 2) &
    (X >= long / 2)
)

mask_racetrack = mask_rectangle | mask_left_circle | mask_right_circle

a = (diameter - neck_width_corrected) / 2
b = 0.0
print(f"Solved a = {a:.3f}, c = {c:.3f}")

y_bottom_curve = a * np.exp(-(X - b) ** 2 / (2 * c ** 2)) - diameter / 2
y_top_curve = -a * np.exp(-(X - b) ** 2 / (2 * c ** 2)) + diameter / 2

mask_peanut = mask_racetrack & (Y >= y_bottom_curve) & (Y <= y_top_curve)

bottom_plane_index = np.isclose(z, 0, atol=1e-8).nonzero()[0][0]
mask_bottom = mask_peanut[:, :, bottom_plane_index]

boundary_points = []
for i, xi in enumerate(x):
    ys = np.where(mask_bottom[i, :])[0]
    if len(ys) > 0:
        y_boundary = y[ys[0]]
        boundary_points.append((xi, y_boundary))
boundary_points = np.array(boundary_points)

center_idx = np.argmin(np.abs(boundary_points[:, 0]))
center_point = boundary_points[center_idx]

left_half = boundary_points[boundary_points[:, 0] < 0]
right_half = boundary_points[boundary_points[:, 0] > 0]

left_point = left_half[np.argmin(left_half[:, 1])]
right_point = right_half[np.argmin(right_half[:, 1])]

if abs(left_point[0]) > abs(right_point[0]):
    symmetric_x = -right_point[0]
    left_point = left_half[np.argmin(np.abs(left_half[:, 0] - symmetric_x))]
else:
    symmetric_x = -left_point[0]
    right_point = right_half[np.argmin(np.abs(right_half[:, 0] - symmetric_x))]

def calculate_relative_angle(center, point, side="left"):
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    angle = np.degrees(np.arctan2(dy, dx))
    if side == "left":
        return 180 - abs(angle)
    else:
        return abs(angle)

angle_left = calculate_relative_angle(center_point, left_point, side="left")
angle_right = calculate_relative_angle(center_point, right_point, side="right")

print(f"Angle 1 (left): {angle_left:.2f} degrees")
print(f"Angle 2 (right): {angle_right:.2f} degrees")

fig, ax = plt.subplots(figsize=(8, 4))

ax.contour(
    x, y,
    mask_bottom.T.astype(float),
    levels=[0.5],
    colors='black',
    linewidths=3.5
)

ax.scatter(*center_point, color='black', s=70, zorder=5, label="key point")
ax.scatter(*left_point, color='black', s=70, zorder=5)
ax.scatter(*right_point, color='black', s=70, zorder=5)

ax.plot(
    [center_point[0], left_point[0]],
    [center_point[1], left_point[1]],
    color='red',
    linewidth=2.8
)
ax.plot(
    [center_point[0], right_point[0]],
    [center_point[1], right_point[1]],
    color='red',
    linewidth=2.8
)

text_offset = 0.3
ax.text(
    (center_point[0] + left_point[0]) / 1.1,
    (center_point[1] + left_point[1]) / 2 + text_offset,
    f"{angle_left:.1f}°",
    color='red',
    fontsize=26,
    fontweight='bold'
)
ax.text(
    (center_point[0] + right_point[0]) / 2.9,
    (center_point[1] + right_point[1]) / 2 + text_offset,
    f"{angle_right:.1f}°",
    color='red',
    fontsize=26,
    fontweight='bold'
)

ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("")
ax.set_ylabel("")

for spine in ax.spines.values():
    spine.set_visible(False)

plt.savefig(
    "slot_angle_measurement.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()
