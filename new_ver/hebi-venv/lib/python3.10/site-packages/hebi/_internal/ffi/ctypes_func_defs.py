from ctypes import CDLL

_hebi_api = None
_hebi_wrapper_api = None

# C API Functions
hebiLookupCreate = None
hebiLookupRelease = None
hebiLookupReset = None
hebiLookupSetLookupFrequencyHz = None
hebiLookupGetLookupFrequencyHz = None
hebiCreateLookupEntryList = None
hebiLookupEntryListGetSize = None
hebiLookupEntryListGetName = None
hebiLookupEntryListGetFamily = None
hebiLookupEntryListGetMacAddress = None
hebiLookupEntryListRelease = None
hebiGroupCreateImitation = None
hebiGroupCreateFromMacs = None
hebiGroupCreateFromNames = None
hebiGroupCreateFromFamily = None
hebiGroupCreateConnectedFromMac = None
hebiGroupCreateConnectedFromName = None
hebiGroupGetSize = None
hebiGroupSendCommandWithAcknowledgement = None
hebiGroupSendCommand = None
hebiGroupSetCommandLifetime = None
hebiGroupGetCommandLifetime = None
hebiGroupSetFeedbackFrequencyHz = None
hebiGroupGetFeedbackFrequencyHz = None
hebiGroupRegisterFeedbackHandler = None
hebiGroupClearFeedbackHandlers = None
hebiGroupSendFeedbackRequest = None
hebiGroupGetNextFeedback = None
hebiGroupRequestInfo = None
hebiGroupStartLog = None
hebiGroupStopLog = None
hebiGroupRelease = None
hebiGroupCommandCreate = None
hebiGroupCommandGetSize = None
hebiGroupCommandReadGains = None
hebiGroupCommandWriteGains = None
hebiGroupCommandCopyGainsFromInfo = None
hebiGroupCommandCopyGainsFromCommand = None
hebiGroupCommandReadSafetyParameters = None
hebiGroupCommandWriteSafetyParameters = None
hebiGroupCommandGetModuleCommand = None
hebiGroupCommandCopy = None
hebiGroupCommandClear = None
hebiGroupCommandRelease = None
hebiGroupFeedbackCreate = None
hebiGroupFeedbackGetSize = None
hebiGroupFeedbackGetModuleFeedback = None
hebiGroupFeedbackCopy = None
hebiGroupFeedbackClear = None
hebiGroupFeedbackRelease = None
hebiGroupInfoCreate = None
hebiGroupInfoGetSize = None
hebiGroupInfoWriteGains = None
hebiGroupInfoWriteSafetyParameters = None
hebiGroupInfoGetModuleInfo = None
hebiGroupInfoCopy = None
hebiGroupInfoClear = None
hebiGroupInfoRelease = None
hebiCommandGetString = None
hebiCommandSetString = None
hebiCommandGetIoLabelString = None
hebiCommandSetIoLabelString = None
hebiCommandGetReference = None
hebiCommandGetSize = None
hebiCommandGetMetadata = None
hebiCommandCopyGainsFromInfo = None
hebiCommandCopyGainsFromCommand = None
hebiFeedbackGetReference = None
hebiFeedbackGetSize = None
hebiFeedbackGetMetadata = None
hebiInfoGetString = None
hebiInfoGetIoLabelString = None
hebiInfoGetReference = None
hebiInfoGetSize = None
hebiInfoGetMetadata = None
hebiRobotModelElementCreateJoint = None
hebiRobotModelElementCreateRigidBody = None
hebiRobotModelElementCreateEndEffector = None
hebiRobotModelElementCreateActuator = None
hebiRobotModelElementCreateBracket = None
hebiRobotModelElementCreateLink = None
hebiRobotModelElementRelease = None
hebiRobotModelImport = None
hebiRobotModelImportBuffer = None
hebiRobotModelGetImportError = None
hebiRobotModelGetImportWarningCount = None
hebiRobotModelGetImportWarning = None
hebiRobotModelCreate = None
hebiRobotModelSetBaseFrame = None
hebiRobotModelGetBaseFrame = None
hebiRobotModelGetNumberOfFrames = None
hebiRobotModelGetNumberOfDoFs = None
hebiRobotModelGetNumberOfElements = None
hebiRobotModelGetElementMetadata = None
hebiRobotModelAdd = None
hebiRobotModelGetForwardKinematics = None
hebiRobotModelGetJacobians = None
hebiRobotModelGetMasses = None
hebiRobotModelGetTreeTopology = None
hebiRobotModelRelease = None
hebiIKCreate = None
hebiIKAddObjectiveEndEffectorPosition = None
hebiIKAddObjectiveFramePosition = None
hebiIKAddObjectiveEndEffectorSO3 = None
hebiIKAddObjectiveFrameSO3 = None
hebiIKAddObjectiveEndEffectorTipAxis = None
hebiIKAddObjectiveFrameTipAxis = None
hebiIKAddConstraintJointAngles = None
hebiIKAddObjectiveCustom = None
hebiIKClearAll = None
hebiIKSolve = None
hebiIKRelease = None
hebiTrajectoryCreateUnconstrainedQp = None
hebiTrajectoryRelease = None
hebiTrajectoryGetDuration = None
hebiTrajectoryGetState = None
hebiLogFileRelease = None
hebiLogFileGetFileName = None
hebiLogFileOpen = None
hebiLogFileGetNumberOfModules = None
hebiLogFileGetNextFeedback = None
hebiStringGetString = None
hebiStringRelease = None
hebiSafetyParametersGetLastError = None
hebiGetLibraryVersion = None
hebiCleanup = None

# Wrapper Functions
hwInitialize = None
hwCommandSetFloat = None
hwCommandSetHighResAngle = None
hwCommandSetEnum = None
hwCommandSetBool = None
hwCommandSetNumberedFloat = None
hwCommandSetIoPin = None
hwCommandSetIoPinInt = None
hwCommandSetIoPinFloat = None
hwCommandSetLed = None
hwCommandSetFlag = None
hwCommandGetFloat = None
hwCommandGetHighResAngle = None
hwCommandGetVector3f = None
hwCommandGetQuaternionf = None
hwCommandGetUInt64 = None
hwCommandGetEnum = None
hwCommandGetBool = None
hwCommandGetNumberedFloat = None
hwCommandGetIoPin = None
hwCommandGetIoPinInt = None
hwCommandGetIoPinFloat = None
hwCommandGetLed = None
hwCommandGetFlag = None
hwCommandHasField = None
hwCommandHasIoPinInt = None
hwCommandHasIoPinFloat = None
hwFeedbackGetFloat = None
hwFeedbackGetHighResAngle = None
hwFeedbackGetVector3f = None
hwFeedbackGetQuaternionf = None
hwFeedbackGetUInt64 = None
hwFeedbackGetEnum = None
hwFeedbackGetBool = None
hwFeedbackGetNumberedFloat = None
hwFeedbackGetIoPin = None
hwFeedbackGetIoPinInt = None
hwFeedbackGetIoPinFloat = None
hwFeedbackGetLed = None
hwFeedbackHasField = None
hwFeedbackHasIoPinInt = None
hwFeedbackHasIoPinFloat = None
hwInfoGetFloat = None
hwInfoGetHighResAngle = None
hwInfoGetFlag = None
hwInfoGetBool = None
hwInfoGetEnum = None
hwInfoGetLed = None
hwInfoHasField = None


def set_api_handle(lib: CDLL):
  """Sets the handle to the HEBI Core C API."""
  global _hebi_api
  _hebi_api = lib

  global hebiLookupCreate
  hebiLookupCreate = lib.hebiLookupCreate
  global hebiLookupRelease
  hebiLookupRelease = lib.hebiLookupRelease
  global hebiLookupReset
  hebiLookupReset = lib.hebiLookupReset
  global hebiLookupSetLookupFrequencyHz
  hebiLookupSetLookupFrequencyHz = lib.hebiLookupSetLookupFrequencyHz
  global hebiLookupGetLookupFrequencyHz
  hebiLookupGetLookupFrequencyHz = lib.hebiLookupGetLookupFrequencyHz
  global hebiCreateLookupEntryList
  hebiCreateLookupEntryList = lib.hebiCreateLookupEntryList
  global hebiLookupEntryListGetSize
  hebiLookupEntryListGetSize = lib.hebiLookupEntryListGetSize
  global hebiLookupEntryListGetName
  hebiLookupEntryListGetName = lib.hebiLookupEntryListGetName
  global hebiLookupEntryListGetFamily
  hebiLookupEntryListGetFamily = lib.hebiLookupEntryListGetFamily
  global hebiLookupEntryListGetMacAddress
  hebiLookupEntryListGetMacAddress = lib.hebiLookupEntryListGetMacAddress
  global hebiLookupEntryListRelease
  hebiLookupEntryListRelease = lib.hebiLookupEntryListRelease
  global hebiGroupCreateImitation
  hebiGroupCreateImitation = lib.hebiGroupCreateImitation
  global hebiGroupCreateFromMacs
  hebiGroupCreateFromMacs = lib.hebiGroupCreateFromMacs
  global hebiGroupCreateFromNames
  hebiGroupCreateFromNames = lib.hebiGroupCreateFromNames
  global hebiGroupCreateFromFamily
  hebiGroupCreateFromFamily = lib.hebiGroupCreateFromFamily
  global hebiGroupCreateConnectedFromMac
  hebiGroupCreateConnectedFromMac = lib.hebiGroupCreateConnectedFromMac
  global hebiGroupCreateConnectedFromName
  hebiGroupCreateConnectedFromName = lib.hebiGroupCreateConnectedFromName
  global hebiGroupGetSize
  hebiGroupGetSize = lib.hebiGroupGetSize
  global hebiGroupSendCommandWithAcknowledgement
  hebiGroupSendCommandWithAcknowledgement = lib.hebiGroupSendCommandWithAcknowledgement
  global hebiGroupSendCommand
  hebiGroupSendCommand = lib.hebiGroupSendCommand
  global hebiGroupSetCommandLifetime
  hebiGroupSetCommandLifetime = lib.hebiGroupSetCommandLifetime
  global hebiGroupGetCommandLifetime
  hebiGroupGetCommandLifetime = lib.hebiGroupGetCommandLifetime
  global hebiGroupSetFeedbackFrequencyHz
  hebiGroupSetFeedbackFrequencyHz = lib.hebiGroupSetFeedbackFrequencyHz
  global hebiGroupGetFeedbackFrequencyHz
  hebiGroupGetFeedbackFrequencyHz = lib.hebiGroupGetFeedbackFrequencyHz
  global hebiGroupRegisterFeedbackHandler
  hebiGroupRegisterFeedbackHandler = lib.hebiGroupRegisterFeedbackHandler
  global hebiGroupClearFeedbackHandlers
  hebiGroupClearFeedbackHandlers = lib.hebiGroupClearFeedbackHandlers
  global hebiGroupSendFeedbackRequest
  hebiGroupSendFeedbackRequest = lib.hebiGroupSendFeedbackRequest
  global hebiGroupGetNextFeedback
  hebiGroupGetNextFeedback = lib.hebiGroupGetNextFeedback
  global hebiGroupRequestInfo
  hebiGroupRequestInfo = lib.hebiGroupRequestInfo
  global hebiGroupStartLog
  hebiGroupStartLog = lib.hebiGroupStartLog
  global hebiGroupStopLog
  hebiGroupStopLog = lib.hebiGroupStopLog
  global hebiGroupRelease
  hebiGroupRelease = lib.hebiGroupRelease
  global hebiGroupCommandCreate
  hebiGroupCommandCreate = lib.hebiGroupCommandCreate
  global hebiGroupCommandGetSize
  hebiGroupCommandGetSize = lib.hebiGroupCommandGetSize
  global hebiGroupCommandReadGains
  hebiGroupCommandReadGains = lib.hebiGroupCommandReadGains
  global hebiGroupCommandWriteGains
  hebiGroupCommandWriteGains = lib.hebiGroupCommandWriteGains
  global hebiGroupCommandCopyGainsFromInfo
  hebiGroupCommandCopyGainsFromInfo = lib.hebiGroupCommandCopyGainsFromInfo
  global hebiGroupCommandCopyGainsFromCommand
  hebiGroupCommandCopyGainsFromCommand = lib.hebiGroupCommandCopyGainsFromCommand
  global hebiGroupCommandReadSafetyParameters
  hebiGroupCommandReadSafetyParameters = lib.hebiGroupCommandReadSafetyParameters
  global hebiGroupCommandWriteSafetyParameters
  hebiGroupCommandWriteSafetyParameters = lib.hebiGroupCommandWriteSafetyParameters
  global hebiGroupCommandGetModuleCommand
  hebiGroupCommandGetModuleCommand = lib.hebiGroupCommandGetModuleCommand
  global hebiGroupCommandCopy
  hebiGroupCommandCopy = lib.hebiGroupCommandCopy
  global hebiGroupCommandClear
  hebiGroupCommandClear = lib.hebiGroupCommandClear
  global hebiGroupCommandRelease
  hebiGroupCommandRelease = lib.hebiGroupCommandRelease
  global hebiGroupFeedbackCreate
  hebiGroupFeedbackCreate = lib.hebiGroupFeedbackCreate
  global hebiGroupFeedbackGetSize
  hebiGroupFeedbackGetSize = lib.hebiGroupFeedbackGetSize
  global hebiGroupFeedbackGetModuleFeedback
  hebiGroupFeedbackGetModuleFeedback = lib.hebiGroupFeedbackGetModuleFeedback
  global hebiGroupFeedbackCopy
  hebiGroupFeedbackCopy = lib.hebiGroupFeedbackCopy
  global hebiGroupFeedbackClear
  hebiGroupFeedbackClear = lib.hebiGroupFeedbackClear
  global hebiGroupFeedbackRelease
  hebiGroupFeedbackRelease = lib.hebiGroupFeedbackRelease
  global hebiGroupInfoCreate
  hebiGroupInfoCreate = lib.hebiGroupInfoCreate
  global hebiGroupInfoGetSize
  hebiGroupInfoGetSize = lib.hebiGroupInfoGetSize
  global hebiGroupInfoWriteGains
  hebiGroupInfoWriteGains = lib.hebiGroupInfoWriteGains
  global hebiGroupInfoWriteSafetyParameters
  hebiGroupInfoWriteSafetyParameters = lib.hebiGroupInfoWriteSafetyParameters
  global hebiGroupInfoGetModuleInfo
  hebiGroupInfoGetModuleInfo = lib.hebiGroupInfoGetModuleInfo
  global hebiGroupInfoCopy
  hebiGroupInfoCopy = lib.hebiGroupInfoCopy
  global hebiGroupInfoClear
  hebiGroupInfoClear = lib.hebiGroupInfoClear
  global hebiGroupInfoRelease
  hebiGroupInfoRelease = lib.hebiGroupInfoRelease
  global hebiCommandGetString
  hebiCommandGetString = lib.hebiCommandGetString
  global hebiCommandSetString
  hebiCommandSetString = lib.hebiCommandSetString
  global hebiCommandGetIoLabelString
  hebiCommandGetIoLabelString = lib.hebiCommandGetIoLabelString
  global hebiCommandSetIoLabelString
  hebiCommandSetIoLabelString = lib.hebiCommandSetIoLabelString
  global hebiCommandGetReference
  hebiCommandGetReference = lib.hebiCommandGetReference
  global hebiCommandGetSize
  hebiCommandGetSize = lib.hebiCommandGetSize
  global hebiCommandGetMetadata
  hebiCommandGetMetadata = lib.hebiCommandGetMetadata
  global hebiCommandCopyGainsFromInfo
  hebiCommandCopyGainsFromInfo = lib.hebiCommandCopyGainsFromInfo
  global hebiCommandCopyGainsFromCommand
  hebiCommandCopyGainsFromCommand = lib.hebiCommandCopyGainsFromCommand
  global hebiFeedbackGetReference
  hebiFeedbackGetReference = lib.hebiFeedbackGetReference
  global hebiFeedbackGetSize
  hebiFeedbackGetSize = lib.hebiFeedbackGetSize
  global hebiFeedbackGetMetadata
  hebiFeedbackGetMetadata = lib.hebiFeedbackGetMetadata
  global hebiInfoGetString
  hebiInfoGetString = lib.hebiInfoGetString
  global hebiInfoGetIoLabelString
  hebiInfoGetIoLabelString = lib.hebiInfoGetIoLabelString
  global hebiInfoGetReference
  hebiInfoGetReference = lib.hebiInfoGetReference
  global hebiInfoGetSize
  hebiInfoGetSize = lib.hebiInfoGetSize
  global hebiInfoGetMetadata
  hebiInfoGetMetadata = lib.hebiInfoGetMetadata
  global hebiRobotModelElementCreateJoint
  hebiRobotModelElementCreateJoint = lib.hebiRobotModelElementCreateJoint
  global hebiRobotModelElementCreateRigidBody
  hebiRobotModelElementCreateRigidBody = lib.hebiRobotModelElementCreateRigidBody
  global hebiRobotModelElementCreateEndEffector
  hebiRobotModelElementCreateEndEffector = lib.hebiRobotModelElementCreateEndEffector
  global hebiRobotModelElementCreateActuator
  hebiRobotModelElementCreateActuator = lib.hebiRobotModelElementCreateActuator
  global hebiRobotModelElementCreateBracket
  hebiRobotModelElementCreateBracket = lib.hebiRobotModelElementCreateBracket
  global hebiRobotModelElementCreateLink
  hebiRobotModelElementCreateLink = lib.hebiRobotModelElementCreateLink
  global hebiRobotModelElementRelease
  hebiRobotModelElementRelease = lib.hebiRobotModelElementRelease
  global hebiRobotModelImport
  hebiRobotModelImport = lib.hebiRobotModelImport
  global hebiRobotModelImportBuffer
  hebiRobotModelImportBuffer = lib.hebiRobotModelImportBuffer
  global hebiRobotModelGetImportError
  hebiRobotModelGetImportError = lib.hebiRobotModelGetImportError
  global hebiRobotModelGetImportWarningCount
  hebiRobotModelGetImportWarningCount = lib.hebiRobotModelGetImportWarningCount
  global hebiRobotModelGetImportWarning
  hebiRobotModelGetImportWarning = lib.hebiRobotModelGetImportWarning
  global hebiRobotModelCreate
  hebiRobotModelCreate = lib.hebiRobotModelCreate
  global hebiRobotModelSetBaseFrame
  hebiRobotModelSetBaseFrame = lib.hebiRobotModelSetBaseFrame
  global hebiRobotModelGetBaseFrame
  hebiRobotModelGetBaseFrame = lib.hebiRobotModelGetBaseFrame
  global hebiRobotModelGetNumberOfFrames
  hebiRobotModelGetNumberOfFrames = lib.hebiRobotModelGetNumberOfFrames
  global hebiRobotModelGetNumberOfDoFs
  hebiRobotModelGetNumberOfDoFs = lib.hebiRobotModelGetNumberOfDoFs
  global hebiRobotModelGetNumberOfElements
  hebiRobotModelGetNumberOfElements = lib.hebiRobotModelGetNumberOfElements
  global hebiRobotModelGetElementMetadata
  hebiRobotModelGetElementMetadata = lib.hebiRobotModelGetElementMetadata
  global hebiRobotModelAdd
  hebiRobotModelAdd = lib.hebiRobotModelAdd
  global hebiRobotModelGetForwardKinematics
  hebiRobotModelGetForwardKinematics = lib.hebiRobotModelGetForwardKinematics
  global hebiRobotModelGetJacobians
  hebiRobotModelGetJacobians = lib.hebiRobotModelGetJacobians
  global hebiRobotModelGetMasses
  hebiRobotModelGetMasses = lib.hebiRobotModelGetMasses
  global hebiRobotModelGetTreeTopology
  hebiRobotModelGetTreeTopology = lib.hebiRobotModelGetTreeTopology
  global hebiRobotModelRelease
  hebiRobotModelRelease = lib.hebiRobotModelRelease
  global hebiIKCreate
  hebiIKCreate = lib.hebiIKCreate
  global hebiIKAddObjectiveEndEffectorPosition
  hebiIKAddObjectiveEndEffectorPosition = lib.hebiIKAddObjectiveEndEffectorPosition
  global hebiIKAddObjectiveFramePosition
  hebiIKAddObjectiveFramePosition = lib.hebiIKAddObjectiveFramePosition
  global hebiIKAddObjectiveEndEffectorSO3
  hebiIKAddObjectiveEndEffectorSO3 = lib.hebiIKAddObjectiveEndEffectorSO3
  global hebiIKAddObjectiveFrameSO3
  hebiIKAddObjectiveFrameSO3 = lib.hebiIKAddObjectiveFrameSO3
  global hebiIKAddObjectiveEndEffectorTipAxis
  hebiIKAddObjectiveEndEffectorTipAxis = lib.hebiIKAddObjectiveEndEffectorTipAxis
  global hebiIKAddObjectiveFrameTipAxis
  hebiIKAddObjectiveFrameTipAxis = lib.hebiIKAddObjectiveFrameTipAxis
  global hebiIKAddConstraintJointAngles
  hebiIKAddConstraintJointAngles = lib.hebiIKAddConstraintJointAngles
  global hebiIKAddObjectiveCustom
  hebiIKAddObjectiveCustom = lib.hebiIKAddObjectiveCustom
  global hebiIKClearAll
  hebiIKClearAll = lib.hebiIKClearAll
  global hebiIKSolve
  hebiIKSolve = lib.hebiIKSolve
  global hebiIKRelease
  hebiIKRelease = lib.hebiIKRelease
  global hebiTrajectoryCreateUnconstrainedQp
  hebiTrajectoryCreateUnconstrainedQp = lib.hebiTrajectoryCreateUnconstrainedQp
  global hebiTrajectoryRelease
  hebiTrajectoryRelease = lib.hebiTrajectoryRelease
  global hebiTrajectoryGetDuration
  hebiTrajectoryGetDuration = lib.hebiTrajectoryGetDuration
  global hebiTrajectoryGetState
  hebiTrajectoryGetState = lib.hebiTrajectoryGetState
  global hebiLogFileRelease
  hebiLogFileRelease = lib.hebiLogFileRelease
  global hebiLogFileGetFileName
  hebiLogFileGetFileName = lib.hebiLogFileGetFileName
  global hebiLogFileOpen
  hebiLogFileOpen = lib.hebiLogFileOpen
  global hebiLogFileGetNumberOfModules
  hebiLogFileGetNumberOfModules = lib.hebiLogFileGetNumberOfModules
  global hebiLogFileGetNextFeedback
  hebiLogFileGetNextFeedback = lib.hebiLogFileGetNextFeedback
  global hebiStringGetString
  hebiStringGetString = lib.hebiStringGetString
  global hebiStringRelease
  hebiStringRelease = lib.hebiStringRelease
  global hebiSafetyParametersGetLastError
  hebiSafetyParametersGetLastError = lib.hebiSafetyParametersGetLastError
  global hebiGetLibraryVersion
  hebiGetLibraryVersion = lib.hebiGetLibraryVersion
  global hebiCleanup
  hebiCleanup = lib.hebiCleanup


def set_api_wrapper_handle(lib: CDLL):
  """Sets the handle to the C wrapper API."""
  global _hebi_wrapper_api
  _hebi_wrapper_api = lib

  global hwInitialize
  hwInitialize = lib.hwInitialize
  global hwCommandSetFloat
  hwCommandSetFloat = lib.hwCommandSetFloat
  global hwCommandSetHighResAngle
  hwCommandSetHighResAngle = lib.hwCommandSetHighResAngle
  global hwCommandSetEnum
  hwCommandSetEnum = lib.hwCommandSetEnum
  global hwCommandSetBool
  hwCommandSetBool = lib.hwCommandSetBool
  global hwCommandSetNumberedFloat
  hwCommandSetNumberedFloat = lib.hwCommandSetNumberedFloat
  global hwCommandSetIoPin
  hwCommandSetIoPin = lib.hwCommandSetIoPin
  global hwCommandSetIoPinInt
  hwCommandSetIoPinInt = lib.hwCommandSetIoPinInt
  global hwCommandSetIoPinFloat
  hwCommandSetIoPinFloat = lib.hwCommandSetIoPinFloat
  global hwCommandSetLed
  hwCommandSetLed = lib.hwCommandSetLed
  global hwCommandSetFlag
  hwCommandSetFlag = lib.hwCommandSetFlag
  global hwCommandGetFloat
  hwCommandGetFloat = lib.hwCommandGetFloat
  global hwCommandGetHighResAngle
  hwCommandGetHighResAngle = lib.hwCommandGetHighResAngle
  global hwCommandGetVector3f
  hwCommandGetVector3f = lib.hwCommandGetVector3f
  global hwCommandGetQuaternionf
  hwCommandGetQuaternionf = lib.hwCommandGetQuaternionf
  global hwCommandGetUInt64
  hwCommandGetUInt64 = lib.hwCommandGetUInt64
  global hwCommandGetEnum
  hwCommandGetEnum = lib.hwCommandGetEnum
  global hwCommandGetBool
  hwCommandGetBool = lib.hwCommandGetBool
  global hwCommandGetNumberedFloat
  hwCommandGetNumberedFloat = lib.hwCommandGetNumberedFloat
  global hwCommandGetIoPin
  hwCommandGetIoPin = lib.hwCommandGetIoPin
  global hwCommandGetIoPinInt
  hwCommandGetIoPinInt = lib.hwCommandGetIoPinInt
  global hwCommandGetIoPinFloat
  hwCommandGetIoPinFloat = lib.hwCommandGetIoPinFloat
  global hwCommandGetLed
  hwCommandGetLed = lib.hwCommandGetLed
  global hwCommandGetFlag
  hwCommandGetFlag = lib.hwCommandGetFlag
  global hwCommandHasField
  hwCommandHasField = lib.hwCommandHasField
  global hwCommandHasIoPinInt
  hwCommandHasIoPinInt = lib.hwCommandHasIoPinInt
  global hwCommandHasIoPinFloat
  hwCommandHasIoPinFloat = lib.hwCommandHasIoPinFloat
  global hwFeedbackGetFloat
  hwFeedbackGetFloat = lib.hwFeedbackGetFloat
  global hwFeedbackGetHighResAngle
  hwFeedbackGetHighResAngle = lib.hwFeedbackGetHighResAngle
  global hwFeedbackGetVector3f
  hwFeedbackGetVector3f = lib.hwFeedbackGetVector3f
  global hwFeedbackGetQuaternionf
  hwFeedbackGetQuaternionf = lib.hwFeedbackGetQuaternionf
  global hwFeedbackGetUInt64
  hwFeedbackGetUInt64 = lib.hwFeedbackGetUInt64
  global hwFeedbackGetEnum
  hwFeedbackGetEnum = lib.hwFeedbackGetEnum
  global hwFeedbackGetBool
  hwFeedbackGetBool = lib.hwFeedbackGetBool
  global hwFeedbackGetNumberedFloat
  hwFeedbackGetNumberedFloat = lib.hwFeedbackGetNumberedFloat
  global hwFeedbackGetIoPin
  hwFeedbackGetIoPin = lib.hwFeedbackGetIoPin
  global hwFeedbackGetIoPinInt
  hwFeedbackGetIoPinInt = lib.hwFeedbackGetIoPinInt
  global hwFeedbackGetIoPinFloat
  hwFeedbackGetIoPinFloat = lib.hwFeedbackGetIoPinFloat
  global hwFeedbackGetLed
  hwFeedbackGetLed = lib.hwFeedbackGetLed
  global hwFeedbackHasField
  hwFeedbackHasField = lib.hwFeedbackHasField
  global hwFeedbackHasIoPinInt
  hwFeedbackHasIoPinInt = lib.hwFeedbackHasIoPinInt
  global hwFeedbackHasIoPinFloat
  hwFeedbackHasIoPinFloat = lib.hwFeedbackHasIoPinFloat
  global hwInfoGetFloat
  hwInfoGetFloat = lib.hwInfoGetFloat
  global hwInfoGetHighResAngle
  hwInfoGetHighResAngle = lib.hwInfoGetHighResAngle
  global hwInfoGetFlag
  hwInfoGetFlag = lib.hwInfoGetFlag
  global hwInfoGetBool
  hwInfoGetBool = lib.hwInfoGetBool
  global hwInfoGetEnum
  hwInfoGetEnum = lib.hwInfoGetEnum
  global hwInfoGetLed
  hwInfoGetLed = lib.hwInfoGetLed
  global hwInfoHasField
  hwInfoHasField = lib.hwInfoHasField
