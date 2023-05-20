import numpy as np
from math import sin, cos


def rotate_transform(thex, they, thez):
    """
    Compute the rotation transformation matrix given rotation angles along x, y, and z axes.

    Args:
        thex (float): Rotation angle along x-axis in radians.
        they (float): Rotation angle along y-axis in radians.
        thez (float): Rotation angle along z-axis in radians.

    Returns:
        numpy.ndarray: Rotation transformation matrix.
    """
    cos_thex, sin_thex = cos(thex), sin(thex)
    cos_they, sin_they = cos(they), sin(they)
    cos_thez, sin_thez = cos(thez), sin(thez)

    Rx = np.array([[1, 0, 0, 0],
                   [0, cos_thex, -sin_thex, 0],
                   [0, sin_thex, cos_thex, 0],
                   [0, 0, 0, 1]])

    Ry = np.array([[cos_they, 0, sin_they, 0],
                   [0, 1, 0, 0],
                   [-sin_they, 0, cos_they, 0],
                   [0, 0, 0, 1]])

    Rz = np.array([[cos_thez, -sin_thez, 0, 0],
                   [sin_thez, cos_thez, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

    return Rx @ Ry @ Rz


def DH_transform(a, alpha, d, theta):
    """
    Compute the Denavit-Hartenberg homogeneous transformation matrix.

    Args:
        a (float): Link length.
        alpha (float): Link twist in radians.
        d (float): Link offset.
        theta (float): Joint angle in radians.

    Returns:
        numpy.ndarray: Denavit-Hartenberg homogeneous transformation matrix.
    """
    cos_theta, sin_theta = cos(theta), sin(theta)
    cos_alpha, sin_alpha = cos(alpha), sin(alpha)

    return np.array([[cos_theta, -sin_theta * cos_alpha, sin_alpha * sin_theta, a * cos_theta],
                     [sin_theta, cos_alpha * cos_theta, -
                         sin_alpha * cos_theta, a * sin_theta],
                     [0, sin_alpha, cos_alpha, d],
                     [0, 0, 0, 1]])
