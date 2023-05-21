# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
#
#  HEBI Core python API - Copyright 2022 HEBI Robotics
#  See https://hebi.us/softwarelicense for license details
#
# -----------------------------------------------------------------------------

from .wrappers import MessageEnumTraits, EnumType, FieldDetails


# CommandFloatField
CommandFloatVelocity = MessageEnumTraits(0, 'CommandFloatVelocity',
                                         field_details=FieldDetails('rad/s', 'velocity', 'Velocity', 'velocity'))
CommandFloatEffort = MessageEnumTraits(1, 'CommandFloatEffort',
                                       field_details=FieldDetails('N*m', 'effort', 'Effort', 'effort'))
CommandFloatPositionKp = MessageEnumTraits(2, 'CommandFloatPositionKp',
                                           field_details=FieldDetails('None', 'positionKp', 'PositionKp', 'position_kp'))
CommandFloatPositionKi = MessageEnumTraits(3, 'CommandFloatPositionKi',
                                           field_details=FieldDetails('None', 'positionKi', 'PositionKi', 'position_ki'))
CommandFloatPositionKd = MessageEnumTraits(4, 'CommandFloatPositionKd',
                                           field_details=FieldDetails('None', 'positionKd', 'PositionKd', 'position_kd'))
CommandFloatPositionFeedForward = MessageEnumTraits(5, 'CommandFloatPositionFeedForward',
                                                    field_details=FieldDetails('None', 'positionFeedForward', 'PositionFeedForward', 'position_feed_forward'))
CommandFloatPositionDeadZone = MessageEnumTraits(6, 'CommandFloatPositionDeadZone',
                                                 field_details=FieldDetails('None', 'positionDeadZone', 'PositionDeadZone', 'position_dead_zone'))
CommandFloatPositionIClamp = MessageEnumTraits(7, 'CommandFloatPositionIClamp',
                                               field_details=FieldDetails('None', 'positionIClamp', 'PositionIClamp', 'position_i_clamp'))
CommandFloatPositionPunch = MessageEnumTraits(8, 'CommandFloatPositionPunch',
                                              field_details=FieldDetails('None', 'positionPunch', 'PositionPunch', 'position_punch'))
CommandFloatPositionMinTarget = MessageEnumTraits(9, 'CommandFloatPositionMinTarget',
                                                  field_details=FieldDetails('None', 'positionMinTarget', 'PositionMinTarget', 'position_min_target'))
CommandFloatPositionMaxTarget = MessageEnumTraits(10, 'CommandFloatPositionMaxTarget',
                                                  field_details=FieldDetails('None', 'positionMaxTarget', 'PositionMaxTarget', 'position_max_target'))
CommandFloatPositionTargetLowpass = MessageEnumTraits(11, 'CommandFloatPositionTargetLowpass',
                                                      field_details=FieldDetails('None', 'positionTargetLowpass', 'PositionTargetLowpass', 'position_target_lowpass'))
CommandFloatPositionMinOutput = MessageEnumTraits(12, 'CommandFloatPositionMinOutput',
                                                  field_details=FieldDetails('None', 'positionMinOutput', 'PositionMinOutput', 'position_min_output'))
CommandFloatPositionMaxOutput = MessageEnumTraits(13, 'CommandFloatPositionMaxOutput',
                                                  field_details=FieldDetails('None', 'positionMaxOutput', 'PositionMaxOutput', 'position_max_output'))
CommandFloatPositionOutputLowpass = MessageEnumTraits(14, 'CommandFloatPositionOutputLowpass',
                                                      field_details=FieldDetails('None', 'positionOutputLowpass', 'PositionOutputLowpass', 'position_output_lowpass'))
CommandFloatVelocityKp = MessageEnumTraits(15, 'CommandFloatVelocityKp',
                                           field_details=FieldDetails('None', 'velocityKp', 'VelocityKp', 'velocity_kp'))
CommandFloatVelocityKi = MessageEnumTraits(16, 'CommandFloatVelocityKi',
                                           field_details=FieldDetails('None', 'velocityKi', 'VelocityKi', 'velocity_ki'))
CommandFloatVelocityKd = MessageEnumTraits(17, 'CommandFloatVelocityKd',
                                           field_details=FieldDetails('None', 'velocityKd', 'VelocityKd', 'velocity_kd'))
CommandFloatVelocityFeedForward = MessageEnumTraits(18, 'CommandFloatVelocityFeedForward',
                                                    field_details=FieldDetails('None', 'velocityFeedForward', 'VelocityFeedForward', 'velocity_feed_forward'))
CommandFloatVelocityDeadZone = MessageEnumTraits(19, 'CommandFloatVelocityDeadZone',
                                                 field_details=FieldDetails('None', 'velocityDeadZone', 'VelocityDeadZone', 'velocity_dead_zone'))
CommandFloatVelocityIClamp = MessageEnumTraits(20, 'CommandFloatVelocityIClamp',
                                               field_details=FieldDetails('None', 'velocityIClamp', 'VelocityIClamp', 'velocity_i_clamp'))
CommandFloatVelocityPunch = MessageEnumTraits(21, 'CommandFloatVelocityPunch',
                                              field_details=FieldDetails('None', 'velocityPunch', 'VelocityPunch', 'velocity_punch'))
CommandFloatVelocityMinTarget = MessageEnumTraits(22, 'CommandFloatVelocityMinTarget',
                                                  field_details=FieldDetails('None', 'velocityMinTarget', 'VelocityMinTarget', 'velocity_min_target'))
CommandFloatVelocityMaxTarget = MessageEnumTraits(23, 'CommandFloatVelocityMaxTarget',
                                                  field_details=FieldDetails('None', 'velocityMaxTarget', 'VelocityMaxTarget', 'velocity_max_target'))
CommandFloatVelocityTargetLowpass = MessageEnumTraits(24, 'CommandFloatVelocityTargetLowpass',
                                                      field_details=FieldDetails('None', 'velocityTargetLowpass', 'VelocityTargetLowpass', 'velocity_target_lowpass'))
CommandFloatVelocityMinOutput = MessageEnumTraits(25, 'CommandFloatVelocityMinOutput',
                                                  field_details=FieldDetails('None', 'velocityMinOutput', 'VelocityMinOutput', 'velocity_min_output'))
CommandFloatVelocityMaxOutput = MessageEnumTraits(26, 'CommandFloatVelocityMaxOutput',
                                                  field_details=FieldDetails('None', 'velocityMaxOutput', 'VelocityMaxOutput', 'velocity_max_output'))
CommandFloatVelocityOutputLowpass = MessageEnumTraits(27, 'CommandFloatVelocityOutputLowpass',
                                                      field_details=FieldDetails('None', 'velocityOutputLowpass', 'VelocityOutputLowpass', 'velocity_output_lowpass'))
CommandFloatEffortKp = MessageEnumTraits(28, 'CommandFloatEffortKp',
                                         field_details=FieldDetails('None', 'effortKp', 'EffortKp', 'effort_kp'))
CommandFloatEffortKi = MessageEnumTraits(29, 'CommandFloatEffortKi',
                                         field_details=FieldDetails('None', 'effortKi', 'EffortKi', 'effort_ki'))
CommandFloatEffortKd = MessageEnumTraits(30, 'CommandFloatEffortKd',
                                         field_details=FieldDetails('None', 'effortKd', 'EffortKd', 'effort_kd'))
CommandFloatEffortFeedForward = MessageEnumTraits(31, 'CommandFloatEffortFeedForward',
                                                  field_details=FieldDetails('None', 'effortFeedForward', 'EffortFeedForward', 'effort_feed_forward'))
CommandFloatEffortDeadZone = MessageEnumTraits(32, 'CommandFloatEffortDeadZone',
                                               field_details=FieldDetails('None', 'effortDeadZone', 'EffortDeadZone', 'effort_dead_zone'))
CommandFloatEffortIClamp = MessageEnumTraits(33, 'CommandFloatEffortIClamp',
                                             field_details=FieldDetails('None', 'effortIClamp', 'EffortIClamp', 'effort_i_clamp'))
CommandFloatEffortPunch = MessageEnumTraits(34, 'CommandFloatEffortPunch',
                                            field_details=FieldDetails('None', 'effortPunch', 'EffortPunch', 'effort_punch'))
CommandFloatEffortMinTarget = MessageEnumTraits(35, 'CommandFloatEffortMinTarget',
                                                field_details=FieldDetails('None', 'effortMinTarget', 'EffortMinTarget', 'effort_min_target'))
CommandFloatEffortMaxTarget = MessageEnumTraits(36, 'CommandFloatEffortMaxTarget',
                                                field_details=FieldDetails('None', 'effortMaxTarget', 'EffortMaxTarget', 'effort_max_target'))
CommandFloatEffortTargetLowpass = MessageEnumTraits(37, 'CommandFloatEffortTargetLowpass',
                                                    field_details=FieldDetails('None', 'effortTargetLowpass', 'EffortTargetLowpass', 'effort_target_lowpass'))
CommandFloatEffortMinOutput = MessageEnumTraits(38, 'CommandFloatEffortMinOutput',
                                                field_details=FieldDetails('None', 'effortMinOutput', 'EffortMinOutput', 'effort_min_output'))
CommandFloatEffortMaxOutput = MessageEnumTraits(39, 'CommandFloatEffortMaxOutput',
                                                field_details=FieldDetails('None', 'effortMaxOutput', 'EffortMaxOutput', 'effort_max_output'))
CommandFloatEffortOutputLowpass = MessageEnumTraits(40, 'CommandFloatEffortOutputLowpass',
                                                    field_details=FieldDetails('None', 'effortOutputLowpass', 'EffortOutputLowpass', 'effort_output_lowpass'))
CommandFloatSpringConstant = MessageEnumTraits(41, 'CommandFloatSpringConstant',
                                               field_details=FieldDetails('N/m', 'springConstant', 'SpringConstant', 'spring_constant'))
CommandFloatReferencePosition = MessageEnumTraits(42, 'CommandFloatReferencePosition',
                                                  field_details=FieldDetails('rad', 'referencePosition', 'ReferencePosition', 'reference_position'))
CommandFloatReferenceEffort = MessageEnumTraits(43, 'CommandFloatReferenceEffort',
                                                field_details=FieldDetails('N*m', 'referenceEffort', 'ReferenceEffort', 'reference_effort'))
CommandFloatVelocityLimitMin = MessageEnumTraits(44, 'CommandFloatVelocityLimitMin',
                                                 field_details=FieldDetails('rad/s', 'velocityLimitMin', 'VelocityLimitMin', 'velocity_limit_min'))
CommandFloatVelocityLimitMax = MessageEnumTraits(45, 'CommandFloatVelocityLimitMax',
                                                 field_details=FieldDetails('rad/s', 'velocityLimitMax', 'VelocityLimitMax', 'velocity_limit_max'))
CommandFloatEffortLimitMin = MessageEnumTraits(46, 'CommandFloatEffortLimitMin',
                                               field_details=FieldDetails('N*m', 'effortLimitMin', 'EffortLimitMin', 'effort_limit_min'))
CommandFloatEffortLimitMax = MessageEnumTraits(47, 'CommandFloatEffortLimitMax',
                                               field_details=FieldDetails('N*m', 'effortLimitMax', 'EffortLimitMax', 'effort_limit_max'))
CommandFloatMotorFocId = MessageEnumTraits(48, 'CommandFloatMotorFocId',
                                               field_details=FieldDetails('???', 'motorFocId', 'MotorFocId', 'motor_foc_id'))
CommandFloatMotorFocIq = MessageEnumTraits(49, 'CommandFloatMotorFocIq',
                                               field_details=FieldDetails('???', 'motorFocIq', 'MotorFocIq', 'motor_foc_iq'))
CommandFloatField = EnumType('CommandFloatField', [CommandFloatVelocity, CommandFloatEffort, CommandFloatPositionKp, CommandFloatPositionKi, CommandFloatPositionKd, CommandFloatPositionFeedForward, CommandFloatPositionDeadZone, CommandFloatPositionIClamp, CommandFloatPositionPunch, CommandFloatPositionMinTarget, CommandFloatPositionMaxTarget, CommandFloatPositionTargetLowpass, CommandFloatPositionMinOutput, CommandFloatPositionMaxOutput, CommandFloatPositionOutputLowpass, CommandFloatVelocityKp, CommandFloatVelocityKi, CommandFloatVelocityKd, CommandFloatVelocityFeedForward, CommandFloatVelocityDeadZone, CommandFloatVelocityIClamp, CommandFloatVelocityPunch, CommandFloatVelocityMinTarget, CommandFloatVelocityMaxTarget,
                             CommandFloatVelocityTargetLowpass, CommandFloatVelocityMinOutput, CommandFloatVelocityMaxOutput, CommandFloatVelocityOutputLowpass, CommandFloatEffortKp, CommandFloatEffortKi, CommandFloatEffortKd, CommandFloatEffortFeedForward, CommandFloatEffortDeadZone, CommandFloatEffortIClamp, CommandFloatEffortPunch, CommandFloatEffortMinTarget, CommandFloatEffortMaxTarget, CommandFloatEffortTargetLowpass, CommandFloatEffortMinOutput, CommandFloatEffortMaxOutput, CommandFloatEffortOutputLowpass, CommandFloatSpringConstant, CommandFloatReferencePosition, CommandFloatReferenceEffort, CommandFloatVelocityLimitMin, CommandFloatVelocityLimitMax, CommandFloatEffortLimitMin, CommandFloatEffortLimitMax, CommandFloatMotorFocId, CommandFloatMotorFocIq])

# CommandHighResAngleField
CommandHighResAnglePosition = MessageEnumTraits(0, 'CommandHighResAnglePosition',
                                                field_details=FieldDetails('rad', 'position', 'Position', 'position'))
CommandHighResAnglePositionLimitMin = MessageEnumTraits(1, 'CommandHighResAnglePositionLimitMin',
                                                        field_details=FieldDetails('rad', 'positionLimitMin', 'PositionLimitMin', 'position_limit_min'))
CommandHighResAnglePositionLimitMax = MessageEnumTraits(2, 'CommandHighResAnglePositionLimitMax',
                                                        field_details=FieldDetails('rad', 'positionLimitMax', 'PositionLimitMax', 'position_limit_max'))
CommandHighResAngleField = EnumType('CommandHighResAngleField', [CommandHighResAnglePosition,
                                    CommandHighResAnglePositionLimitMin, CommandHighResAnglePositionLimitMax, ])


# CommandEnumField
CommandEnumControlStrategy = MessageEnumTraits(0, 'CommandEnumControlStrategy',
                                               field_details=FieldDetails('None', 'controlStrategy', 'ControlStrategy', 'control_strategy'))
CommandEnumMstopStrategy = MessageEnumTraits(1, 'CommandEnumMstopStrategy',
                                             field_details=FieldDetails('None', 'mstopStrategy', 'MstopStrategy', 'mstop_strategy'))
CommandEnumMinPositionLimitStrategy = MessageEnumTraits(2, 'CommandEnumMinPositionLimitStrategy', field_details=FieldDetails(
    'None', 'minPositionLimitStrategy', 'MinPositionLimitStrategy', 'min_position_limit_strategy'))
CommandEnumMaxPositionLimitStrategy = MessageEnumTraits(3, 'CommandEnumMaxPositionLimitStrategy', field_details=FieldDetails(
    'None', 'maxPositionLimitStrategy', 'MaxPositionLimitStrategy', 'max_position_limit_strategy'))
CommandEnumField = EnumType('CommandEnumField', [CommandEnumControlStrategy, CommandEnumMstopStrategy,
                            CommandEnumMinPositionLimitStrategy, CommandEnumMaxPositionLimitStrategy, ])

# CommandBoolField
CommandBoolPositionDOnError = MessageEnumTraits(0, 'CommandBoolPositionDOnError',
                                                field_details=FieldDetails('None', 'positionDOnError', 'PositionDOnError', 'position_d_on_error'))
CommandBoolVelocityDOnError = MessageEnumTraits(1, 'CommandBoolVelocityDOnError',
                                                field_details=FieldDetails('None', 'velocityDOnError', 'VelocityDOnError', 'velocity_d_on_error'))
CommandBoolEffortDOnError = MessageEnumTraits(2, 'CommandBoolEffortDOnError',
                                              field_details=FieldDetails('None', 'effortDOnError', 'EffortDOnError', 'effort_d_on_error'))
CommandBoolAccelIncludesGravity = MessageEnumTraits(3, 'CommandBoolAccelIncludesGravity',
                                                    field_details=FieldDetails('None', 'accelIncludesGravity', 'AccelIncludesGravity', 'accel_includes_gravity'))
CommandBoolField = EnumType('CommandBoolField', [CommandBoolPositionDOnError, CommandBoolVelocityDOnError,
                            CommandBoolEffortDOnError, CommandBoolAccelIncludesGravity, ])

# CommandNumberedFloatField
CommandNumberedFloatDebug = MessageEnumTraits(0, 'CommandNumberedFloatDebug',
                                              field_details=None)
CommandNumberedFloatField = EnumType('CommandNumberedFloatField', [CommandNumberedFloatDebug, ])

# CommandIoBankField
CommandIoBankA = MessageEnumTraits(0, 'CommandIoBankA', field_details=None)
CommandIoBankB = MessageEnumTraits(1, 'CommandIoBankB', field_details=None)
CommandIoBankC = MessageEnumTraits(2, 'CommandIoBankC', field_details=None)
CommandIoBankD = MessageEnumTraits(3, 'CommandIoBankD', field_details=None)
CommandIoBankE = MessageEnumTraits(4, 'CommandIoBankE', field_details=None)
CommandIoBankF = MessageEnumTraits(5, 'CommandIoBankF', field_details=None)
CommandIoBankField = EnumType('CommandIoBankField', [CommandIoBankA, CommandIoBankB,
                              CommandIoBankC, CommandIoBankD, CommandIoBankE, CommandIoBankF, ])

# CommandLedField
CommandLedLed = MessageEnumTraits(0, 'CommandLedLed',
                                  field_details=FieldDetails('None', 'led', 'Led', 'led'))
CommandLedField = EnumType('CommandLedField', [CommandLedLed, ])

# CommandStringField
CommandStringName = MessageEnumTraits(0, 'CommandStringName',
                                      not_bcastable_reason='Cannot set same name for all modules in a group.',
                                      field_details=None)
CommandStringFamily = MessageEnumTraits(1, 'CommandStringFamily',
                                        field_details=None)
CommandStringAppendLog = MessageEnumTraits(2, 'CommandStringAppendLog',
                                           field_details=None)
CommandStringField = EnumType('CommandStringField', [CommandStringName, CommandStringFamily, CommandStringAppendLog, ])

# CommandFlagField
CommandFlagSaveCurrentSettings = MessageEnumTraits(0, 'CommandFlagSaveCurrentSettings',
                                                   field_details=FieldDetails('None', 'saveCurrentSettings', 'SaveCurrentSettings', 'save_current_settings'))
CommandFlagReset = MessageEnumTraits(1, 'CommandFlagReset',
                                     field_details=FieldDetails('None', 'reset', 'Reset', 'reset'))
CommandFlagBoot = MessageEnumTraits(2, 'CommandFlagBoot',
                                    field_details=FieldDetails('None', 'boot', 'Boot', 'boot'))
CommandFlagStopBoot = MessageEnumTraits(3, 'CommandFlagStopBoot',
                                        field_details=FieldDetails('None', 'stopBoot', 'StopBoot', 'stop_boot'))
CommandFlagClearLog = MessageEnumTraits(4, 'CommandFlagClearLog',
                                        field_details=FieldDetails('None', 'clearLog', 'ClearLog', 'clear_log'))
CommandFlagField = EnumType('CommandFlagField', [CommandFlagSaveCurrentSettings, CommandFlagReset,
                            CommandFlagBoot, CommandFlagStopBoot, CommandFlagClearLog, ])
# FeedbackFloatField
FeedbackFloatBoardTemperature = MessageEnumTraits(0, 'FeedbackFloatBoardTemperature',
                                                  field_details=FieldDetails('C', 'boardTemperature', 'BoardTemperature', 'board_temperature'))
FeedbackFloatProcessorTemperature = MessageEnumTraits(1, 'FeedbackFloatProcessorTemperature',
                                                      field_details=FieldDetails('C', 'processorTemperature', 'ProcessorTemperature', 'processor_temperature'))
FeedbackFloatVoltage = MessageEnumTraits(2, 'FeedbackFloatVoltage',
                                         field_details=FieldDetails('V', 'voltage', 'Voltage', 'voltage'))
FeedbackFloatVelocity = MessageEnumTraits(3, 'FeedbackFloatVelocity',
                                          field_details=FieldDetails('rad/s', 'velocity', 'Velocity', 'velocity'))
FeedbackFloatEffort = MessageEnumTraits(4, 'FeedbackFloatEffort',
                                        field_details=FieldDetails('N*m', 'effort', 'Effort', 'effort'))
FeedbackFloatVelocityCommand = MessageEnumTraits(5, 'FeedbackFloatVelocityCommand',
                                                 field_details=FieldDetails('rad/s', 'velocityCommand', 'VelocityCommand', 'velocity_command'))
FeedbackFloatEffortCommand = MessageEnumTraits(6, 'FeedbackFloatEffortCommand',
                                               field_details=FieldDetails('N*m', 'effortCommand', 'EffortCommand', 'effort_command'))
FeedbackFloatDeflection = MessageEnumTraits(7, 'FeedbackFloatDeflection',
                                            field_details=FieldDetails('rad', 'deflection', 'Deflection', 'deflection'))
FeedbackFloatDeflectionVelocity = MessageEnumTraits(8, 'FeedbackFloatDeflectionVelocity',
                                                    field_details=FieldDetails('rad/s', 'deflectionVelocity', 'DeflectionVelocity', 'deflection_velocity'))
FeedbackFloatMotorVelocity = MessageEnumTraits(9, 'FeedbackFloatMotorVelocity',
                                               field_details=FieldDetails('rad/s', 'motorVelocity', 'MotorVelocity', 'motor_velocity'))
FeedbackFloatMotorCurrent = MessageEnumTraits(10, 'FeedbackFloatMotorCurrent',
                                              field_details=FieldDetails('A', 'motorCurrent', 'MotorCurrent', 'motor_current'))
FeedbackFloatMotorSensorTemperature = MessageEnumTraits(11, 'FeedbackFloatMotorSensorTemperature',
                                                        field_details=FieldDetails('C', 'motorSensorTemperature', 'MotorSensorTemperature', 'motor_sensor_temperature'))
FeedbackFloatMotorWindingCurrent = MessageEnumTraits(12, 'FeedbackFloatMotorWindingCurrent',
                                                     field_details=FieldDetails('A', 'motorWindingCurrent', 'MotorWindingCurrent', 'motor_winding_current'))
FeedbackFloatMotorWindingTemperature = MessageEnumTraits(13, 'FeedbackFloatMotorWindingTemperature',
                                                         field_details=FieldDetails('C', 'motorWindingTemperature', 'MotorWindingTemperature', 'motor_winding_temperature'))
FeedbackFloatMotorHousingTemperature = MessageEnumTraits(14, 'FeedbackFloatMotorHousingTemperature',
                                                         field_details=FieldDetails('C', 'motorHousingTemperature', 'MotorHousingTemperature', 'motor_housing_temperature'))
FeedbackFloatBatteryLevel = MessageEnumTraits(15, 'FeedbackFloatBatteryLevel',
                                              field_details=FieldDetails('None', 'batteryLevel', 'BatteryLevel', 'battery_level'))
FeedbackFloatPwmCommand = MessageEnumTraits(16, 'FeedbackFloatPwmCommand',
                                            field_details=FieldDetails('None', 'pwmCommand', 'PwmCommand', 'pwm_command'))
FeedbackFloatInnerEffortCommand = MessageEnumTraits(17, 'FeedbackFloatInnerEffortCommand',
                                                    field_details=FieldDetails('???', 'innerEffortCommand', 'InnerEffortCommand', 'inner_effort_command'))
FeedbackFloatMotorWindingVoltage = MessageEnumTraits(18, 'FeedbackFloatMotorWindingVoltage',
                                                     field_details=FieldDetails('???', 'motorWindingVoltage', 'MotorWindingVoltage', 'motor_winding_voltage'))
FeedbackFloatMotorPhaseCurrentA = MessageEnumTraits(19, 'FeedbackFloatMotorPhaseCurrentA',
                                                    field_details=FieldDetails('???', 'motorPhaseCurrentA', 'MotorPhaseCurrentA', 'motor_phase_current_a'))
FeedbackFloatMotorPhaseCurrentB = MessageEnumTraits(20, 'FeedbackFloatMotorPhaseCurrentB',
                                                    field_details=FieldDetails('???', 'motorPhaseCurrentB', 'MotorPhaseCurrentB', 'motor_phase_current_b'))
FeedbackFloatMotorPhaseCurrentC = MessageEnumTraits(21, 'FeedbackFloatMotorPhaseCurrentC',
                                                    field_details=FieldDetails('???', 'motorPhaseCurrentC', 'MotorPhaseCurrentC', 'motor_phase_current_c'))
FeedbackFloatMotorPhaseVoltageA = MessageEnumTraits(22, 'FeedbackFloatMotorPhaseVoltageA',
                                                    field_details=FieldDetails('???', 'motorPhaseVoltageA', 'MotorPhaseVoltageA', 'motor_phase_voltage_a'))
FeedbackFloatMotorPhaseVoltageB = MessageEnumTraits(23, 'FeedbackFloatMotorPhaseVoltageB',
                                                    field_details=FieldDetails('???', 'motorPhaseVoltageB', 'MotorPhaseVoltageB', 'motor_phase_voltage_b'))
FeedbackFloatMotorPhaseVoltageC = MessageEnumTraits(24, 'FeedbackFloatMotorPhaseVoltageC',
                                                    field_details=FieldDetails('???', 'motorPhaseVoltageC', 'MotorPhaseVoltageC', 'motor_phase_voltage_c'))
FeedbackFloatMotorPhaseDutyCycleA = MessageEnumTraits(25, 'FeedbackFloatMotorPhaseDutyCycleA',
                                                      field_details=FieldDetails('???', 'motorPhaseDutyCycleA', 'MotorPhaseDutyCycleA', 'motor_phase_duty_cycle_a'))
FeedbackFloatMotorPhaseDutyCycleB = MessageEnumTraits(26, 'FeedbackFloatMotorPhaseDutyCycleB',
                                                      field_details=FieldDetails('???', 'motorPhaseDutyCycleB', 'MotorPhaseDutyCycleB', 'motor_phase_duty_cycle_b'))
FeedbackFloatMotorPhaseDutyCycleC = MessageEnumTraits(27, 'FeedbackFloatMotorPhaseDutyCycleC',
                                                      field_details=FieldDetails('???', 'motorPhaseDutyCycleC', 'MotorPhaseDutyCycleC', 'motor_phase_duty_cycle_c'))
FeedbackFloatMotorFocId = MessageEnumTraits(28, 'FeedbackFloatMotorFocId',
                                            field_details=FieldDetails('None', 'motorFocId', 'MotorFocId', 'motor_foc_id'))
FeedbackFloatMotorFocIq = MessageEnumTraits(29, 'FeedbackFloatMotorFocIq',
                                            field_details=FieldDetails('None', 'motorFocIq', 'MotorFocIq', 'motor_foc_iq'))
FeedbackFloatMotorFocIdCommand = MessageEnumTraits(30, 'FeedbackFloatMotorFocIdCommand',
                                                   field_details=FieldDetails('None', 'motorFocIdCommand', 'MotorFocIdCommand', 'motor_foc_id_command'))
FeedbackFloatMotorFocIqCommand = MessageEnumTraits(31, 'FeedbackFloatMotorFocIqCommand',
                                                   field_details=FieldDetails('None', 'motorFocIqCommand', 'MotorFocIqCommand', 'motor_foc_iq_command'))
FeedbackFloatField = EnumType('FeedbackFloatField', [FeedbackFloatBoardTemperature, FeedbackFloatProcessorTemperature, FeedbackFloatVoltage, FeedbackFloatVelocity, FeedbackFloatEffort, FeedbackFloatVelocityCommand, FeedbackFloatEffortCommand, FeedbackFloatDeflection,
                              FeedbackFloatDeflectionVelocity, FeedbackFloatMotorVelocity, FeedbackFloatMotorCurrent, FeedbackFloatMotorSensorTemperature, FeedbackFloatMotorWindingCurrent, FeedbackFloatMotorWindingTemperature, FeedbackFloatMotorHousingTemperature, FeedbackFloatBatteryLevel, FeedbackFloatPwmCommand, FeedbackFloatInnerEffortCommand, FeedbackFloatMotorWindingVoltage,
                              FeedbackFloatMotorPhaseCurrentA, FeedbackFloatMotorPhaseCurrentB, FeedbackFloatMotorPhaseCurrentC, FeedbackFloatMotorPhaseVoltageA, FeedbackFloatMotorPhaseVoltageB, FeedbackFloatMotorPhaseVoltageC, FeedbackFloatMotorPhaseDutyCycleA, FeedbackFloatMotorPhaseDutyCycleB, FeedbackFloatMotorPhaseDutyCycleC, FeedbackFloatMotorFocId, FeedbackFloatMotorFocIq, FeedbackFloatMotorFocIdCommand, FeedbackFloatMotorFocIqCommand, ])

# FeedbackHighResAngleField
FeedbackHighResAnglePosition = MessageEnumTraits(0, 'FeedbackHighResAnglePosition',
                                                 field_details=FieldDetails('rad', 'position', 'Position', 'position'))
FeedbackHighResAnglePositionCommand = MessageEnumTraits(1, 'FeedbackHighResAnglePositionCommand',
                                                        field_details=FieldDetails('rad', 'positionCommand', 'PositionCommand', 'position_command'))
FeedbackHighResAngleMotorPosition = MessageEnumTraits(2, 'FeedbackHighResAngleMotorPosition',
                                                      field_details=FieldDetails('rad', 'motorPosition', 'MotorPosition', 'motor_position'))
FeedbackHighResAngleField = EnumType('FeedbackHighResAngleField', [
                                     FeedbackHighResAnglePosition, FeedbackHighResAnglePositionCommand, FeedbackHighResAngleMotorPosition, ])

# FeedbackVector3fField
FeedbackVector3fAccelerometer = MessageEnumTraits(0, 'FeedbackVector3fAccelerometer',
                                                  field_details=None)
FeedbackVector3fGyro = MessageEnumTraits(1, 'FeedbackVector3fGyro',
                                         field_details=None)
FeedbackVector3fArPosition = MessageEnumTraits(2, 'FeedbackVector3fArPosition',
                                               field_details=None)
FeedbackVector3fField = EnumType('FeedbackVector3fField', [FeedbackVector3fAccelerometer, FeedbackVector3fGyro, FeedbackVector3fArPosition, ])

# FeedbackQuaternionfField
FeedbackQuaternionfOrientation = MessageEnumTraits(0, 'FeedbackQuaternionfOrientation',
                                                   field_details=None)
FeedbackQuaternionfArOrientation = MessageEnumTraits(1, 'FeedbackQuaternionfArOrientation',
                                                     field_details=None)
FeedbackQuaternionfField = EnumType('FeedbackQuaternionfField', [FeedbackQuaternionfOrientation, FeedbackQuaternionfArOrientation, ])

# FeedbackUInt64Field
FeedbackUInt64SequenceNumber = MessageEnumTraits(0, 'FeedbackUInt64SequenceNumber',
                                                 field_details=FieldDetails('None', 'sequenceNumber', 'SequenceNumber', 'sequence_number'))
FeedbackUInt64ReceiveTime = MessageEnumTraits(1, 'FeedbackUInt64ReceiveTime',
                                              field_details=FieldDetails('us', 'receiveTime', 'ReceiveTime', 'receive_time'))
FeedbackUInt64TransmitTime = MessageEnumTraits(2, 'FeedbackUInt64TransmitTime',
                                               field_details=FieldDetails('us', 'transmitTime', 'TransmitTime', 'transmit_time'))
FeedbackUInt64HardwareReceiveTime = MessageEnumTraits(3, 'FeedbackUInt64HardwareReceiveTime',
                                                      field_details=FieldDetails('us', 'hardwareReceiveTime', 'HardwareReceiveTime', 'hardware_receive_time'))
FeedbackUInt64HardwareTransmitTime = MessageEnumTraits(4, 'FeedbackUInt64HardwareTransmitTime',
                                                       field_details=FieldDetails('us', 'hardwareTransmitTime', 'HardwareTransmitTime', 'hardware_transmit_time'))
FeedbackUInt64SenderId = MessageEnumTraits(5, 'FeedbackUInt64SenderId',
                                           field_details=FieldDetails('None', 'senderId', 'SenderId', 'sender_id'))
FeedbackUInt64RxSequenceNumber = MessageEnumTraits(6, 'FeedbackUInt64RxSequenceNumber',
                                                   field_details=FieldDetails('None', 'rxSequenceNumber', 'RxSequenceNumber', 'rx_sequence_number'))
FeedbackUInt64Field = EnumType('FeedbackUInt64Field', [FeedbackUInt64SequenceNumber, FeedbackUInt64ReceiveTime,
                               FeedbackUInt64TransmitTime, FeedbackUInt64HardwareReceiveTime, FeedbackUInt64HardwareTransmitTime, FeedbackUInt64SenderId, FeedbackUInt64RxSequenceNumber])

# FeedbackEnumField
FeedbackEnumTemperatureState = MessageEnumTraits(0, 'FeedbackEnumTemperatureState',
                                                 field_details=FieldDetails('None', 'temperatureState', 'TemperatureState', 'temperature_state'))
FeedbackEnumMstopState = MessageEnumTraits(1, 'FeedbackEnumMstopState',
                                           field_details=FieldDetails('None', 'mstopState', 'MstopState', 'mstop_state'))
FeedbackEnumPositionLimitState = MessageEnumTraits(2, 'FeedbackEnumPositionLimitState',
                                                   field_details=FieldDetails('None', 'positionLimitState', 'PositionLimitState', 'position_limit_state'))
FeedbackEnumVelocityLimitState = MessageEnumTraits(3, 'FeedbackEnumVelocityLimitState',
                                                   field_details=FieldDetails('None', 'velocityLimitState', 'VelocityLimitState', 'velocity_limit_state'))
FeedbackEnumEffortLimitState = MessageEnumTraits(4, 'FeedbackEnumEffortLimitState',
                                                 field_details=FieldDetails('None', 'effortLimitState', 'EffortLimitState', 'effort_limit_state'))
FeedbackEnumCommandLifetimeState = MessageEnumTraits(5, 'FeedbackEnumCommandLifetimeState',
                                                     field_details=FieldDetails('None', 'commandLifetimeState', 'CommandLifetimeState', 'command_lifetime_state'))
FeedbackEnumArQuality = MessageEnumTraits(6, 'FeedbackEnumArQuality',
                                          field_details=FieldDetails('None', 'arQuality', 'ArQuality', 'ar_quality'))
FeedbackEnumMotorHallState = MessageEnumTraits(7, 'FeedbackEnumMotorHallState',
                                               field_details=FieldDetails('None', 'motorHallState', 'MotorHallState', 'motor_hall_state'))
FeedbackEnumField = EnumType('FeedbackEnumField', [FeedbackEnumTemperatureState, FeedbackEnumMstopState, FeedbackEnumPositionLimitState,
                             FeedbackEnumVelocityLimitState, FeedbackEnumEffortLimitState, FeedbackEnumCommandLifetimeState, FeedbackEnumArQuality, FeedbackEnumMotorHallState, ])


# FeedbackNumberedFloatField
FeedbackNumberedFloatDebug = MessageEnumTraits(0, 'FeedbackNumberedFloatDebug',
                                               field_details=None)
FeedbackNumberedFloatField = EnumType('FeedbackNumberedFloatField', [FeedbackNumberedFloatDebug, ])

# FeedbackIoBankField
FeedbackIoBankA = MessageEnumTraits(0, 'FeedbackIoBankA', field_details=None)
FeedbackIoBankB = MessageEnumTraits(1, 'FeedbackIoBankB', field_details=None)
FeedbackIoBankC = MessageEnumTraits(2, 'FeedbackIoBankC', field_details=None)
FeedbackIoBankD = MessageEnumTraits(3, 'FeedbackIoBankD', field_details=None)
FeedbackIoBankE = MessageEnumTraits(4, 'FeedbackIoBankE', field_details=None)
FeedbackIoBankF = MessageEnumTraits(5, 'FeedbackIoBankF', field_details=None)
FeedbackIoBankField = EnumType('FeedbackIoBankField', [FeedbackIoBankA, FeedbackIoBankB,
                               FeedbackIoBankC, FeedbackIoBankD, FeedbackIoBankE, FeedbackIoBankF, ])

# FeedbackLedField
FeedbackLedLed = MessageEnumTraits(0, 'FeedbackLedLed',
                                   field_details=FieldDetails('None', 'led', 'Led', 'led'))
FeedbackLedField = EnumType('FeedbackLedField', [FeedbackLedLed, ])


# InfoFloatField
InfoFloatPositionKp = MessageEnumTraits(0, 'InfoFloatPositionKp',
                                        field_details=FieldDetails('None', 'positionKp', 'PositionKp', 'position_kp'))
InfoFloatPositionKi = MessageEnumTraits(1, 'InfoFloatPositionKi',
                                        field_details=FieldDetails('None', 'positionKi', 'PositionKi', 'position_ki'))
InfoFloatPositionKd = MessageEnumTraits(2, 'InfoFloatPositionKd',
                                        field_details=FieldDetails('None', 'positionKd', 'PositionKd', 'position_kd'))
InfoFloatPositionFeedForward = MessageEnumTraits(3, 'InfoFloatPositionFeedForward',
                                                 field_details=FieldDetails('None', 'positionFeedForward', 'PositionFeedForward', 'position_feed_forward'))
InfoFloatPositionDeadZone = MessageEnumTraits(4, 'InfoFloatPositionDeadZone',
                                              field_details=FieldDetails('None', 'positionDeadZone', 'PositionDeadZone', 'position_dead_zone'))
InfoFloatPositionIClamp = MessageEnumTraits(5, 'InfoFloatPositionIClamp',
                                            field_details=FieldDetails('None', 'positionIClamp', 'PositionIClamp', 'position_i_clamp'))
InfoFloatPositionPunch = MessageEnumTraits(6, 'InfoFloatPositionPunch',
                                           field_details=FieldDetails('None', 'positionPunch', 'PositionPunch', 'position_punch'))
InfoFloatPositionMinTarget = MessageEnumTraits(7, 'InfoFloatPositionMinTarget',
                                               field_details=FieldDetails('None', 'positionMinTarget', 'PositionMinTarget', 'position_min_target'))
InfoFloatPositionMaxTarget = MessageEnumTraits(8, 'InfoFloatPositionMaxTarget',
                                               field_details=FieldDetails('None', 'positionMaxTarget', 'PositionMaxTarget', 'position_max_target'))
InfoFloatPositionTargetLowpass = MessageEnumTraits(9, 'InfoFloatPositionTargetLowpass',
                                                   field_details=FieldDetails('None', 'positionTargetLowpass', 'PositionTargetLowpass', 'position_target_lowpass'))
InfoFloatPositionMinOutput = MessageEnumTraits(10, 'InfoFloatPositionMinOutput',
                                               field_details=FieldDetails('None', 'positionMinOutput', 'PositionMinOutput', 'position_min_output'))
InfoFloatPositionMaxOutput = MessageEnumTraits(11, 'InfoFloatPositionMaxOutput',
                                               field_details=FieldDetails('None', 'positionMaxOutput', 'PositionMaxOutput', 'position_max_output'))
InfoFloatPositionOutputLowpass = MessageEnumTraits(12, 'InfoFloatPositionOutputLowpass',
                                                   field_details=FieldDetails('None', 'positionOutputLowpass', 'PositionOutputLowpass', 'position_output_lowpass'))
InfoFloatVelocityKp = MessageEnumTraits(13, 'InfoFloatVelocityKp',
                                        field_details=FieldDetails('None', 'velocityKp', 'VelocityKp', 'velocity_kp'))
InfoFloatVelocityKi = MessageEnumTraits(14, 'InfoFloatVelocityKi',
                                        field_details=FieldDetails('None', 'velocityKi', 'VelocityKi', 'velocity_ki'))
InfoFloatVelocityKd = MessageEnumTraits(15, 'InfoFloatVelocityKd',
                                        field_details=FieldDetails('None', 'velocityKd', 'VelocityKd', 'velocity_kd'))
InfoFloatVelocityFeedForward = MessageEnumTraits(16, 'InfoFloatVelocityFeedForward',
                                                 field_details=FieldDetails('None', 'velocityFeedForward', 'VelocityFeedForward', 'velocity_feed_forward'))
InfoFloatVelocityDeadZone = MessageEnumTraits(17, 'InfoFloatVelocityDeadZone',
                                              field_details=FieldDetails('None', 'velocityDeadZone', 'VelocityDeadZone', 'velocity_dead_zone'))
InfoFloatVelocityIClamp = MessageEnumTraits(18, 'InfoFloatVelocityIClamp',
                                            field_details=FieldDetails('None', 'velocityIClamp', 'VelocityIClamp', 'velocity_i_clamp'))
InfoFloatVelocityPunch = MessageEnumTraits(19, 'InfoFloatVelocityPunch',
                                           field_details=FieldDetails('None', 'velocityPunch', 'VelocityPunch', 'velocity_punch'))
InfoFloatVelocityMinTarget = MessageEnumTraits(20, 'InfoFloatVelocityMinTarget',
                                               field_details=FieldDetails('None', 'velocityMinTarget', 'VelocityMinTarget', 'velocity_min_target'))
InfoFloatVelocityMaxTarget = MessageEnumTraits(21, 'InfoFloatVelocityMaxTarget',
                                               field_details=FieldDetails('None', 'velocityMaxTarget', 'VelocityMaxTarget', 'velocity_max_target'))
InfoFloatVelocityTargetLowpass = MessageEnumTraits(22, 'InfoFloatVelocityTargetLowpass',
                                                   field_details=FieldDetails('None', 'velocityTargetLowpass', 'VelocityTargetLowpass', 'velocity_target_lowpass'))
InfoFloatVelocityMinOutput = MessageEnumTraits(23, 'InfoFloatVelocityMinOutput',
                                               field_details=FieldDetails('None', 'velocityMinOutput', 'VelocityMinOutput', 'velocity_min_output'))
InfoFloatVelocityMaxOutput = MessageEnumTraits(24, 'InfoFloatVelocityMaxOutput',
                                               field_details=FieldDetails('None', 'velocityMaxOutput', 'VelocityMaxOutput', 'velocity_max_output'))
InfoFloatVelocityOutputLowpass = MessageEnumTraits(25, 'InfoFloatVelocityOutputLowpass',
                                                   field_details=FieldDetails('None', 'velocityOutputLowpass', 'VelocityOutputLowpass', 'velocity_output_lowpass'))
InfoFloatEffortKp = MessageEnumTraits(26, 'InfoFloatEffortKp',
                                      field_details=FieldDetails('None', 'effortKp', 'EffortKp', 'effort_kp'))
InfoFloatEffortKi = MessageEnumTraits(27, 'InfoFloatEffortKi',
                                      field_details=FieldDetails('None', 'effortKi', 'EffortKi', 'effort_ki'))
InfoFloatEffortKd = MessageEnumTraits(28, 'InfoFloatEffortKd',
                                      field_details=FieldDetails('None', 'effortKd', 'EffortKd', 'effort_kd'))
InfoFloatEffortFeedForward = MessageEnumTraits(29, 'InfoFloatEffortFeedForward',
                                               field_details=FieldDetails('None', 'effortFeedForward', 'EffortFeedForward', 'effort_feed_forward'))
InfoFloatEffortDeadZone = MessageEnumTraits(30, 'InfoFloatEffortDeadZone',
                                            field_details=FieldDetails('None', 'effortDeadZone', 'EffortDeadZone', 'effort_dead_zone'))
InfoFloatEffortIClamp = MessageEnumTraits(31, 'InfoFloatEffortIClamp',
                                          field_details=FieldDetails('None', 'effortIClamp', 'EffortIClamp', 'effort_i_clamp'))
InfoFloatEffortPunch = MessageEnumTraits(32, 'InfoFloatEffortPunch',
                                         field_details=FieldDetails('None', 'effortPunch', 'EffortPunch', 'effort_punch'))
InfoFloatEffortMinTarget = MessageEnumTraits(33, 'InfoFloatEffortMinTarget',
                                             field_details=FieldDetails('None', 'effortMinTarget', 'EffortMinTarget', 'effort_min_target'))
InfoFloatEffortMaxTarget = MessageEnumTraits(34, 'InfoFloatEffortMaxTarget',
                                             field_details=FieldDetails('None', 'effortMaxTarget', 'EffortMaxTarget', 'effort_max_target'))
InfoFloatEffortTargetLowpass = MessageEnumTraits(35, 'InfoFloatEffortTargetLowpass',
                                                 field_details=FieldDetails('None', 'effortTargetLowpass', 'EffortTargetLowpass', 'effort_target_lowpass'))
InfoFloatEffortMinOutput = MessageEnumTraits(36, 'InfoFloatEffortMinOutput',
                                             field_details=FieldDetails('None', 'effortMinOutput', 'EffortMinOutput', 'effort_min_output'))
InfoFloatEffortMaxOutput = MessageEnumTraits(37, 'InfoFloatEffortMaxOutput',
                                             field_details=FieldDetails('None', 'effortMaxOutput', 'EffortMaxOutput', 'effort_max_output'))
InfoFloatEffortOutputLowpass = MessageEnumTraits(38, 'InfoFloatEffortOutputLowpass',
                                                 field_details=FieldDetails('None', 'effortOutputLowpass', 'EffortOutputLowpass', 'effort_output_lowpass'))
InfoFloatSpringConstant = MessageEnumTraits(39, 'InfoFloatSpringConstant',
                                            field_details=FieldDetails('N/m', 'springConstant', 'SpringConstant', 'spring_constant'))
InfoFloatVelocityLimitMin = MessageEnumTraits(40, 'InfoFloatVelocityLimitMin',
                                              field_details=FieldDetails('rad/s', 'velocityLimitMin', 'VelocityLimitMin', 'velocity_limit_min'))
InfoFloatVelocityLimitMax = MessageEnumTraits(41, 'InfoFloatVelocityLimitMax',
                                              field_details=FieldDetails('rad/s', 'velocityLimitMax', 'VelocityLimitMax', 'velocity_limit_max'))
InfoFloatEffortLimitMin = MessageEnumTraits(42, 'InfoFloatEffortLimitMin',
                                            field_details=FieldDetails('N*m', 'effortLimitMin', 'EffortLimitMin', 'effort_limit_min'))
InfoFloatEffortLimitMax = MessageEnumTraits(43, 'InfoFloatEffortLimitMax',
                                            field_details=FieldDetails('N*m', 'effortLimitMax', 'EffortLimitMax', 'effort_limit_max'))
InfoFloatField = EnumType('InfoFloatField', [InfoFloatPositionKp, InfoFloatPositionKi, InfoFloatPositionKd, InfoFloatPositionFeedForward, InfoFloatPositionDeadZone, InfoFloatPositionIClamp, InfoFloatPositionPunch, InfoFloatPositionMinTarget, InfoFloatPositionMaxTarget, InfoFloatPositionTargetLowpass, InfoFloatPositionMinOutput, InfoFloatPositionMaxOutput, InfoFloatPositionOutputLowpass, InfoFloatVelocityKp, InfoFloatVelocityKi, InfoFloatVelocityKd, InfoFloatVelocityFeedForward, InfoFloatVelocityDeadZone, InfoFloatVelocityIClamp, InfoFloatVelocityPunch, InfoFloatVelocityMinTarget,
                          InfoFloatVelocityMaxTarget, InfoFloatVelocityTargetLowpass, InfoFloatVelocityMinOutput, InfoFloatVelocityMaxOutput, InfoFloatVelocityOutputLowpass, InfoFloatEffortKp, InfoFloatEffortKi, InfoFloatEffortKd, InfoFloatEffortFeedForward, InfoFloatEffortDeadZone, InfoFloatEffortIClamp, InfoFloatEffortPunch, InfoFloatEffortMinTarget, InfoFloatEffortMaxTarget, InfoFloatEffortTargetLowpass, InfoFloatEffortMinOutput, InfoFloatEffortMaxOutput, InfoFloatEffortOutputLowpass, InfoFloatSpringConstant, InfoFloatVelocityLimitMin, InfoFloatVelocityLimitMax, InfoFloatEffortLimitMin, InfoFloatEffortLimitMax, ])

# InfoHighResAngleField
InfoHighResAnglePositionLimitMin = MessageEnumTraits(0, 'InfoHighResAnglePositionLimitMin',
                                                     field_details=FieldDetails('rad', 'positionLimitMin', 'PositionLimitMin', 'position_limit_min'))
InfoHighResAnglePositionLimitMax = MessageEnumTraits(1, 'InfoHighResAnglePositionLimitMax',
                                                     field_details=FieldDetails('rad', 'positionLimitMax', 'PositionLimitMax', 'position_limit_max'))
InfoHighResAngleField = EnumType('InfoHighResAngleField', [InfoHighResAnglePositionLimitMin, InfoHighResAnglePositionLimitMax, ])


# InfoEnumField
InfoEnumControlStrategy = MessageEnumTraits(0, 'InfoEnumControlStrategy',
                                            field_details=FieldDetails('None', 'controlStrategy', 'ControlStrategy', 'control_strategy'))
InfoEnumCalibrationState = MessageEnumTraits(1, 'InfoEnumCalibrationState',
                                             field_details=FieldDetails('None', 'calibrationState', 'CalibrationState', 'calibration_state'))
InfoEnumMstopStrategy = MessageEnumTraits(2, 'InfoEnumMstopStrategy',
                                          field_details=FieldDetails('None', 'mstopStrategy', 'MstopStrategy', 'mstop_strategy'))
InfoEnumMinPositionLimitStrategy = MessageEnumTraits(3, 'InfoEnumMinPositionLimitStrategy', field_details=FieldDetails(
    'None', 'minPositionLimitStrategy', 'MinPositionLimitStrategy', 'min_position_limit_strategy'))
InfoEnumMaxPositionLimitStrategy = MessageEnumTraits(4, 'InfoEnumMaxPositionLimitStrategy', field_details=FieldDetails(
    'None', 'maxPositionLimitStrategy', 'MaxPositionLimitStrategy', 'max_position_limit_strategy'))
InfoEnumField = EnumType('InfoEnumField', [InfoEnumControlStrategy, InfoEnumCalibrationState,
                         InfoEnumMstopStrategy, InfoEnumMinPositionLimitStrategy, InfoEnumMaxPositionLimitStrategy, ])

# InfoBoolField
InfoBoolPositionDOnError = MessageEnumTraits(0, 'InfoBoolPositionDOnError', field_details=FieldDetails(
    'None', 'positionDOnError', 'PositionDOnError', 'position_d_on_error'))
InfoBoolVelocityDOnError = MessageEnumTraits(1, 'InfoBoolVelocityDOnError', field_details=FieldDetails(
    'None', 'velocityDOnError', 'VelocityDOnError', 'velocity_d_on_error'))
InfoBoolEffortDOnError = MessageEnumTraits(2, 'InfoBoolEffortDOnError', field_details=FieldDetails('None',
                                           'effortDOnError', 'EffortDOnError', 'effort_d_on_error'))
InfoBoolAccelIncludesGravity = MessageEnumTraits(3, 'InfoBoolAccelIncludesGravity', field_details=FieldDetails(
    'None', 'accelIncludesGravity', 'AccelIncludesGravity', 'accel_includes_gravity'))
InfoBoolField = EnumType('InfoBoolField', [InfoBoolPositionDOnError, InfoBoolVelocityDOnError,
                         InfoBoolEffortDOnError, InfoBoolAccelIncludesGravity, ])

# InfoIoBankField
InfoIoBankA = MessageEnumTraits(0, 'InfoIoBankA', field_details=None)
InfoIoBankB = MessageEnumTraits(1, 'InfoIoBankB', field_details=None)
InfoIoBankC = MessageEnumTraits(2, 'InfoIoBankC', field_details=None)
InfoIoBankD = MessageEnumTraits(3, 'InfoIoBankD', field_details=None)
InfoIoBankE = MessageEnumTraits(4, 'InfoIoBankE', field_details=None)
InfoIoBankF = MessageEnumTraits(5, 'InfoIoBankF', field_details=None)
InfoIoBankField = EnumType('InfoIoBankField', [InfoIoBankA, InfoIoBankB, InfoIoBankC, InfoIoBankD, InfoIoBankE, InfoIoBankF, ])

# InfoLedField
InfoLedLed = MessageEnumTraits(0, 'InfoLedLed', field_details=FieldDetails('None', 'led', 'Led', 'led'))
InfoLedField = EnumType('InfoLedField', [InfoLedLed, ])

# InfoStringField
InfoStringName = MessageEnumTraits(0, 'InfoStringName', not_bcastable_reason='Cannot set same name for all modules in a group.', field_details=None)
InfoStringFamily = MessageEnumTraits(1, 'InfoStringFamily', field_details=None)
InfoStringSerial = MessageEnumTraits(2, 'InfoStringSerial', field_details=None)
InfoStringField = EnumType('InfoStringField', [InfoStringName, InfoStringFamily, InfoStringSerial, ])

# InfoFlagField
InfoFlagSaveCurrentSettings = MessageEnumTraits(0, 'InfoFlagSaveCurrentSettings', field_details=FieldDetails(
    'None', 'saveCurrentSettings', 'SaveCurrentSettings', 'save_current_settings'))
InfoFlagField = EnumType('InfoFlagField', [InfoFlagSaveCurrentSettings, ])
