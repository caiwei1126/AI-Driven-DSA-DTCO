import numpy as np


def extract_cd(phi_vals, x, y, L0, L0_nm, ccd_offset_nm=0.0):
    z_index = -2
    phi_vals_top = phi_vals[:, :, z_index]
    y_index = np.argmin(np.abs(y))
    phi_vals_y0 = phi_vals[:, y_index, :]

    y0_max_indices = np.argsort(phi_vals_y0.flatten())[-2:]
    y0_max_indices = np.unravel_index(y0_max_indices, phi_vals_y0.shape)
    max1_y0 = y0_max_indices[0][0], y0_max_indices[1][0]
    max2_y0 = y0_max_indices[0][1], y0_max_indices[1][1]
    x1 = x[max1_y0[0]]
    x2 = x[max2_y0[0]]
    ccd_nm = (np.abs(x1) + np.abs(x2)) / L0 * L0_nm - ccd_offset_nm
    max1_value = phi_vals_y0[max1_y0[0], max1_y0[1]]
    max2_value = phi_vals_y0[max2_y0[0], max2_y0[1]]

    y1 = y[max1_y0[1]]

    cd_x_nm = np.nan
    y_values_greater_than_0_5_max1 = np.where(phi_vals_top[max1_y0[0], :] > 0.39)[0]
    if len(y_values_greater_than_0_5_max1) > 0:
        y_coordinates = y[y_values_greater_than_0_5_max1]
        cd_x_nm = (np.abs(y_coordinates[-1] - y_coordinates[0])) / L0 * L0_nm
    else:
        y_coordinates = []

    cd_y_nm = np.nan
    x_values_greater_than_0_5_max1 = np.where((phi_vals_top[:, max1_y0[1]] > 0.33) & (x < 0))[0]
    if len(x_values_greater_than_0_5_max1) > 0:
        x_coordinates = x[x_values_greater_than_0_5_max1]
        cd_y_nm = (np.abs(-x_coordinates[0] + x_coordinates[-1])) / L0 * L0_nm
    else:
        x_coordinates = []

    return {
        "CCD_nm": ccd_nm,
        "CD_X_nm": cd_x_nm,
        "CD_Y_nm": cd_y_nm,
        "peak_1_x": x1,
        "peak_2_x": x2,
        "peak_1_value": max1_value,
        "peak_2_value": max2_value,
        "y_max1": y1,
        "y_coordinates": y_coordinates,
        "x_coordinates": x_coordinates,
    }


def extract_ccd_from_line(phi_vals, x, y, L0, L0_nm):
    z_index = -2 if (phi_vals.shape[2] >= 2) else -1
    phi_vals_top = phi_vals[:, :, z_index]

    y_index = np.argmin(np.abs(y))
    phi_vals_line = phi_vals_top[:, y_index]

    if phi_vals_line.size < 2:
        return np.nan, phi_vals_line
    two_max_idx = np.argpartition(phi_vals_line, -2)[-2:]
    two_max_idx.sort()
    x1, x2 = x[two_max_idx[0]], x[two_max_idx[1]]
    x_difference = (np.abs(x1) + np.abs(x2)) / L0 * L0_nm
    return x_difference, phi_vals_line
