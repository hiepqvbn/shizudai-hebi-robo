# -*- coding: utf-8 -*-
# ------------------------------------------------------------------------------
#
#  HEBI Core python API - Copyright 2018 HEBI Robotics
#  See https://hebi.us/softwarelicense for license details
#
# ------------------------------------------------------------------------------


import ctypes as _ctypes
import numpy as np
from ._internal import math_utils as _math_utils
from ._internal.trajectory import Trajectory as _Trajectory
from ._internal.ffi import ctypes_func_defs as api

from ._internal.ffi.ctypes_utils import pointer_offset, c_double_p

import typing
if typing.TYPE_CHECKING:
  import numpy.typing as npt


def _check_dims_2d(arr, name, waypoints, joints):
  shape = arr.shape
  shape_expected = (joints, waypoints)
  if shape != shape_expected:
    raise ValueError("Invalid dimensionality of {} matrix (expected {}, got {})".format(name, shape_expected, shape))


def create_trajectory(time: 'npt.ArrayLike', position: 'npt.ArrayLike', velocity: 'npt.ArrayLike | None' = None, acceleration: 'npt.ArrayLike | None' = None):
  """Creates a smooth trajectory through a set of waypoints (position velocity
  and accelerations defined at particular times). This trajectory wrapper
  object can create multi-dimensional trajectories (i.e., multiple joints
  moving together using the same time reference).

  :param time: A vector of desired times at which to reach each
               waypoint; this must be defined
               (and not ``None`` or ``nan`` for any element).
  :type time:  list, numpy.ndarray

  :param position: A matrix of waypoint joint positions (in SI units). The
                   number of rows should be equal to the number of joints,
                   and the number of columns equal to the number of waypoints.
                   Any elements that are ``None`` or ``nan`` will be considered
                   free parameters when solving for a trajectory.
                   Values of ``+/-inf`` are not allowed.
  :type position:  list, numpy.ndarray, ctypes.Array

  :param velocity: An optional matrix of velocity constraints at the
                   corresponding waypoints; should either be ``None``
                   or matching the size of the positions matrix.
                   Any elements that are ``None`` or ``nan`` will be considered
                   free parameters when solving for a trajectory.
                   Values of ``+/-inf`` are not allowed.
  :type velocity:  NoneType, list, numpy.ndarray, ctypes.Array

  :param acceleration: An optional matrix of acceleration constraints at
                       the corresponding waypoints; should either be ``None``
                       or matching the size of the positions matrix.
                       Any elements that are ``None`` or ``nan`` will be considered
                       free parameters when solving for a trajectory.
                       Values of ``+/-inf`` are not allowed.
  :type acceleration:  NoneType, list, numpy.ndarray, ctypes.Array

  :return: The trajectory. This will never be ``None``.
  :rtype: Trajectory

  :raises ValueError: If dimensionality or size of any
                      input parameters are invalid.
  :raises RuntimeError: If trajectory could not be created.
  """
  if time is None:
    raise ValueError("time cannot be None")
  if position is None:
    raise ValueError("position cannot be None")

  time = np.asarray(time, np.float64)
  position = np.asarray(position, np.float64)
  # reshape 1D vector to 1xn 2darray
  if len(position.shape) == 1:
    position = position.reshape((1, -1))
  joints: int = position.shape[0]
  waypoints: int = position.shape[1]

  pointer_stride = waypoints * 8
  shape_checker = lambda arr, name: _check_dims_2d(arr, name, waypoints, joints)

  if time.size != waypoints:
    raise ValueError('length of time vector must be equal to number of waypoints (time:{} != waypoints:{})'.format(time.size, waypoints))

  if not _math_utils.is_finite(time):
    raise ValueError('time vector must have all finite values')

  if velocity is None:
    velocity = np.full(position.shape, np.nan)
    velocity[:, 0] = 0
    velocity[:, -1] = 0

  velocity = np.asarray(velocity, np.float64)
  if len(velocity.shape) == 1:
    velocity = velocity.reshape((1, -1))

  shape_checker(velocity, 'velocity')
  velocity_c = velocity.ctypes.data_as(c_double_p)
  get_vel_offset = lambda i: pointer_offset(velocity_c, i * pointer_stride)

  if acceleration is None:
    acceleration = np.full(position.shape, np.nan)
    acceleration[:, 0] = 0
    acceleration[:, -1] = 0

  acceleration = np.asarray(acceleration, np.float64)
  if len(acceleration.shape) == 1:
    acceleration = acceleration.reshape((1, -1))

  shape_checker(acceleration, 'acceleration')
  acceleration_c = acceleration.ctypes.data_as(c_double_p)
  get_acc_offset = lambda i: pointer_offset(acceleration_c, i * pointer_stride)

  time_c = time.ctypes.data_as(c_double_p)
  position_c = position.ctypes.data_as(c_double_p)
  trajectories = [None] * joints

  for i in range(0, joints):
    pos_offset = pointer_offset(position_c, i * pointer_stride)
    vel_offset = get_vel_offset(i)
    acc_offset = get_acc_offset(i)
    c_trajectory = api.hebiTrajectoryCreateUnconstrainedQp(waypoints, pos_offset, vel_offset, acc_offset, time_c)

    if not c_trajectory:
      raise RuntimeError('Could not create trajectory')
    trajectories[i] = c_trajectory

  return _Trajectory(trajectories, time.copy(), waypoints)
