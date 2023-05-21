from ._message_enums import *
from .wrappers import EnumTraits, EnumType

# Status Codes
StatusSuccess = EnumTraits(0, 'StatusSuccess')
StatusInvalidArgument = EnumTraits(1, 'StatusInvalidArgument')
StatusBufferTooSmall = EnumTraits(2, 'StatusBufferTooSmall')
StatusValueNotSet = EnumTraits(3, 'StatusValueNotSet')
StatusFailure = EnumTraits(4, 'StatusFailure')
StatusArgumentOutOfRange = EnumTraits(5, 'StatusArgumentOutOfRange')
StatusCode = EnumType('StatusCode', [StatusSuccess, StatusInvalidArgument, StatusBufferTooSmall,
                      StatusValueNotSet, StatusFailure, StatusArgumentOutOfRange])

# Frame Types
FrameTypeCenterOfMass = EnumTraits(0, 'FrameTypeCenterOfMass')
FrameTypeOutput = EnumTraits(1, 'FrameTypeOutput')
FrameTypeEndEffector = EnumTraits(2, 'FrameTypeEndEffector')
FrameTypeInput = EnumTraits(3, 'FrameTypeInput')
FrameType = EnumType('FrameType', [FrameTypeCenterOfMass, FrameTypeOutput, FrameTypeEndEffector, FrameTypeInput])

# Robot Model meta data
RobotModelElementTypeOther = EnumTraits(0, 'RobotModelElementTypeOther')
RobotModelElementTypeActuator = EnumTraits(1, 'RobotModelElementTypeActuator')
RobotModelElementTypeBracket = EnumTraits(2, 'RobotModelElementTypeBracket')
RobotModelElementTypeJoint = EnumTraits(3, 'RobotModelElementTypeJoint')
RobotModelElementTypeLink = EnumTraits(4, 'RobotModelElementTypeLink')
RobotModelElementTypeRigidBody = EnumTraits(5, 'RobotModelElementTypeRigidBody')
RobotModelElementTypeEndEffector = EnumTraits(6, 'RobotModelElementTypeEndEffector')
RobotModelElementType = EnumType('RobotModelElementType', [RobotModelElementTypeOther, RobotModelElementTypeActuator, RobotModelElementTypeBracket,
                                 RobotModelElementTypeJoint, RobotModelElementTypeLink, RobotModelElementTypeRigidBody, RobotModelElementTypeEndEffector])

# Types of motion allowed by joints
JointTypeRotationX = EnumTraits(0, 'JointTypeRotationX')
JointTypeRotationY = EnumTraits(1, 'JointTypeRotationY')
JointTypeRotationZ = EnumTraits(2, 'JointTypeRotationZ')
JointTypeTranslationX = EnumTraits(3, 'JointTypeTranslationX')
JointTypeTranslationY = EnumTraits(4, 'JointTypeTranslationY')
JointTypeTranslationZ = EnumTraits(5, 'JointTypeTranslationZ')
JointType = EnumType('JointType', [JointTypeRotationX, JointTypeRotationY, JointTypeRotationZ,
                     JointTypeTranslationX, JointTypeTranslationY, JointTypeTranslationZ])

# Actuator Types
ActuatorTypeX5_1 = EnumTraits(0, 'ActuatorTypeX5_1')
ActuatorTypeX5_4 = EnumTraits(1, 'ActuatorTypeX5_4')
ActuatorTypeX5_9 = EnumTraits(2, 'ActuatorTypeX5_9')
ActuatorTypeX8_3 = EnumTraits(3, 'ActuatorTypeX8_3')
ActuatorTypeX8_9 = EnumTraits(4, 'ActuatorTypeX8_9')
ActuatorTypeX8_16 = EnumTraits(5, 'ActuatorTypeX8_16')
ActuatorTypeR8_3 = EnumTraits(6, 'ActuatorTypeR8_16')
ActuatorTypeR8_9 = EnumTraits(7, 'ActuatorTypeR8_16')
ActuatorTypeR8_16 = EnumTraits(8, 'ActuatorTypeR8_16')
ActuatorType = EnumType('ActuatorType', [ActuatorTypeX5_1, ActuatorTypeX5_4, ActuatorTypeX5_9, ActuatorTypeX8_3,
                        ActuatorTypeX8_9, ActuatorTypeX8_16, ActuatorTypeR8_3, ActuatorTypeR8_9, ActuatorTypeR8_16])

# Link Types
LinkTypeX5 = EnumTraits(0, 'LinkTypeX5')
LinkTypeR8 = EnumTraits(1, 'LinkTypeR8')
LinkType = EnumType('LinkType', [LinkTypeX5, LinkTypeR8])

# Link Input Types
LinkInputTypeRightAngle = EnumTraits(0, 'LinkInputTypeRightAngle')
LinkInputTypeInline = EnumTraits(1, 'LinkInputTypeInline')
LinkInputType = EnumType('LinkInputType', [LinkInputTypeRightAngle, LinkInputTypeInline])

# Link Output Types
LinkOutputTypeRightAngle = EnumTraits(0, 'LinkOutputTypeRightAngle')
LinkOutputTypeInline = EnumTraits(1, 'LinkOutputTypeInline')
LinkOutputType = EnumType('LinkOutputType', [LinkOutputTypeRightAngle, LinkOutputTypeInline])

# Bracket Types
BracketTypeX5LightLeft = EnumTraits(0, 'BracketTypeX5LightLeft')
BracketTypeX5LightRight = EnumTraits(1, 'BracketTypeX5LightRight')
BracketTypeX5HeavyLeftInside = EnumTraits(2, 'BracketTypeX5HeavyLeftInside')
BracketTypeX5HeavyLeftOutside = EnumTraits(3, 'BracketTypeX5HeavyLeftOutside')
BracketTypeX5HeavyRightInside = EnumTraits(4, 'BracketTypeX5HeavyRightInside')
BracketTypeX5HeavyRightOutside = EnumTraits(5, 'BracketTypeX5HeavyRightOutside')
BracketTypeR8LightLeft = EnumTraits(6, 'BracketTypeR8LightLeft')
BracketTypeR8LightRight = EnumTraits(7, 'BracketTypeR8LightRight')
BracketTypeR8HeavyLeftInside = EnumTraits(8, 'BracketTypeR8HeavyLeftInside')
BracketTypeR8HeavyLeftOutside = EnumTraits(9, 'BracketTypeR8HeavyLeftOutside')
BracketTypeR8HeavyRightInside = EnumTraits(10, 'BracketTypeR8HeavyRightInside')
BracketTypeR8HeavyRightOutside = EnumTraits(11, 'BracketTypeR8HeavyRightOutside')
BracketType = EnumType('BracketType', [BracketTypeX5LightLeft, BracketTypeX5LightRight, BracketTypeX5HeavyLeftInside, BracketTypeX5HeavyLeftOutside, BracketTypeX5HeavyRightInside, BracketTypeX5HeavyRightOutside,
                       BracketTypeR8LightLeft, BracketTypeR8LightRight, BracketTypeR8HeavyLeftInside, BracketTypeR8HeavyLeftOutside, BracketTypeR8HeavyRightInside, BracketTypeR8HeavyRightOutside])

EndEffectorTypeCustom = EnumTraits(0, 'EndEffectorTypeCustom')
EndEffectorTypeX5Parallel = EnumTraits(1, 'EndEffectorTypeX5Parallel')
EndEffectorTypeR8Parallel = EnumTraits(2, 'EndEffectorTypeR8Parallel')
EndEffectorType = EnumType('EndEffectorType', [EndEffectorTypeCustom, EndEffectorTypeX5Parallel, EndEffectorTypeR8Parallel])

MatrixOrderingRowMajor = EnumTraits(0, 'MatrixOrderingRowMajor')
MatrixOrderingColumnMajor = EnumTraits(1, 'MatrixOrderingColumnMajor')
MatrixOrdering = EnumType('MatrixOrdering', [MatrixOrderingRowMajor, MatrixOrderingColumnMajor])

# Generated enums for all message field enums
