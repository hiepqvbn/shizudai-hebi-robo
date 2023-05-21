# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#  HEBI Core python API - Copyright 2017-2019 HEBI Robotics
#  See https://hebi.us/softwarelicense for license details
#
# -----------------------------------------------------------------------------

from .ffi import enums as _enums

import typing
if typing.TYPE_CHECKING:
  from .ffi.wrappers import FieldDetails, NumberedFieldDetails


_feedback_scalars = [
    _enums.FeedbackFloatBoardTemperature,
    _enums.FeedbackFloatProcessorTemperature,
    _enums.FeedbackFloatVoltage,
    _enums.FeedbackFloatVelocity,
    _enums.FeedbackFloatEffort,
    _enums.FeedbackFloatVelocityCommand,
    _enums.FeedbackFloatEffortCommand,
    _enums.FeedbackFloatDeflection,
    _enums.FeedbackFloatDeflectionVelocity,
    _enums.FeedbackFloatMotorVelocity,
    _enums.FeedbackFloatMotorCurrent,
    _enums.FeedbackFloatMotorSensorTemperature,
    _enums.FeedbackFloatMotorWindingCurrent,
    _enums.FeedbackFloatMotorWindingTemperature,
    _enums.FeedbackFloatMotorHousingTemperature,
    _enums.FeedbackFloatBatteryLevel,
    _enums.FeedbackFloatPwmCommand,
    _enums.FeedbackHighResAnglePosition,
    _enums.FeedbackHighResAnglePositionCommand,
    _enums.FeedbackHighResAngleMotorPosition,
    _enums.FeedbackUInt64SequenceNumber,
    _enums.FeedbackUInt64ReceiveTime,
    _enums.FeedbackUInt64TransmitTime,
    _enums.FeedbackUInt64HardwareReceiveTime,
    _enums.FeedbackUInt64HardwareTransmitTime,
    _enums.FeedbackUInt64SenderId,
    _enums.FeedbackEnumTemperatureState,
    _enums.FeedbackEnumMstopState,
    _enums.FeedbackEnumPositionLimitState,
    _enums.FeedbackEnumVelocityLimitState,
    _enums.FeedbackEnumEffortLimitState,
    _enums.FeedbackEnumCommandLifetimeState,
    _enums.FeedbackEnumArQuality,
    _enums.FeedbackIoBankA,
    _enums.FeedbackIoBankB,
    _enums.FeedbackIoBankC,
    _enums.FeedbackIoBankD,
    _enums.FeedbackIoBankE,
    _enums.FeedbackIoBankF]


_feedback_scalars_map: 'dict[str, FieldDetails | NumberedFieldDetails.Subfield]' = {}


def __add_fbk_scalars_field(field: 'FieldDetails | NumberedFieldDetails.Subfield'):
  for alias in field.aliases:
    _feedback_scalars_map[alias] = field


def __populate_fbk_scalars_map():
  for entry in _feedback_scalars:

    field = entry.field_details

    if field is None:
      # TODO: THIS IS TEMPORARY. FIX THIS BY DEFINING FIELD_DETAILS FOR ALL FIELDS ABOVE
      continue

    if isinstance(field, FieldDetails):
      # Will be an instance of `MessageEnumTraits`
      __add_fbk_scalars_field(field)
    else:
      for sub_field in field.scalars.values():
        # Unspecified class type: will have all functionality of `MessageEnumTraits`
        __add_fbk_scalars_field(sub_field)


__populate_fbk_scalars_map()


def get_field_info(field_name: str):
  """Get the info object representing the given field name.

  The field binder is a lambda which accepts a group feedback instance and returns the input field name

  :param field_name:
  :return:
  """
  if field_name not in _feedback_scalars_map:
    raise KeyError(field_name)

  return _feedback_scalars_map[field_name]
