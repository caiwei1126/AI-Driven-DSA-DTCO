import numpy as np

f = 0.28
RG = 1.0
WALL = 0.2 * RG
L0 = 4.8 * RG
GRID_SIZE = 0.2
HEIGHT = 2 * L0

PHI_S_WALL = 0.011
PHI_S_BOTTOM = 0.204
LAMBDA_R_BOTTOM = 3.29
LAMBDA_R_SIDE_WALL = 2.05

INIT_PHI_LOW = 0.275
INIT_PHI_HIGH = 0.325


def build_template(diameter_factor, length_factor):
    diameter = diameter_factor * L0
    length = length_factor * L0
    long = length - diameter
    diameter_box = diameter + 2 * WALL
    length_box = length + 2 * WALL

    x_points = int(np.round(length_box / GRID_SIZE))
    y_points = int(np.round(diameter_box / GRID_SIZE))
    z_points = int(np.round(HEIGHT / GRID_SIZE))

    x = np.linspace(-length_box / 2, length_box / 2, x_points)
    y = np.linspace(-diameter_box / 2, diameter_box / 2, y_points)
    z = np.linspace(0, HEIGHT, z_points)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    radius = diameter / 2
    mask_rectangle = (X >= -long / 2) & (X <= long / 2) & (np.abs(Y) <= radius)
    mask_left_circle = ((X + long / 2) ** 2 + Y ** 2 <= radius ** 2) & (X <= -long / 2)
    mask_right_circle = ((X - long / 2) ** 2 + Y ** 2 <= radius ** 2) & (X >= long / 2)
    mask = mask_rectangle | mask_left_circle | mask_right_circle

    distance_to_left = np.sqrt((X + long / 2) ** 2 + Y ** 2)
    distance_to_right = np.sqrt((X - long / 2) ** 2 + Y ** 2)
    distance_to_rect_wall = np.abs(Y)

    side_left = mask_left_circle & (np.abs(distance_to_left - radius) <= (radius / x_points * 4))
    side_right = mask_right_circle & (np.abs(distance_to_right - radius) <= (radius / x_points * 4))
    side_rectangle = mask_rectangle & (
        np.abs(distance_to_rect_wall - diameter / 2) <= (diameter / y_points * 2)
    )
    side_wall_condition = side_left | side_right | side_rectangle

    Lambda_r = np.zeros_like(X)
    bottom_condition = mask & (Z == 0)
    Lambda_r[bottom_condition] = LAMBDA_R_BOTTOM
    Lambda_r[side_wall_condition] = LAMBDA_R_SIDE_WALL

    return {
        "diameter": diameter,
        "length": length,
        "long": long,
        "diameter_box": diameter_box,
        "length_box": length_box,
        "height": HEIGHT,
        "radius": radius,
        "x": x,
        "y": y,
        "z": z,
        "X": X,
        "Y": Y,
        "Z": Z,
        "x_points": x_points,
        "y_points": y_points,
        "z_points": z_points,
        "dl": diameter_box / (y_points - 1),
        "mask": mask,
        "bottom_condition": bottom_condition,
        "side_wall_condition": side_wall_condition,
        "Lambda_r": Lambda_r,
    }


def build_initial_field(template):
    mask = template["mask"]
    phi_vals = np.zeros_like(mask, dtype=np.float64)
    delta_phi_vals = np.full_like(mask, 2 * PHI_S_WALL - 1, dtype=np.float64)
    phi_vals[mask] = np.random.uniform(INIT_PHI_LOW, INIT_PHI_HIGH, size=np.sum(mask))
    delta_phi_vals[mask] = 2 * phi_vals[mask] - 1
    return phi_vals, delta_phi_vals
