# Adapted from wuji-hand-teleop at 647801345a6a27dec5cbf56280ce63bb8b2f6a32.
# Copyright (c) 2025 Wuji Technology Co., Ltd. See LICENSE.

"""C feedback structures shared with the Marvin SDK."""

from ctypes import Structure, c_char, c_float, c_int, c_short, c_ubyte


# Define StateCtr structure
class StateCtr(Structure):
    _fields_ = [
        ("m_CurState", c_int),  # * Current state */ ArmState
        ("m_CmdState", c_int),  # * Command state */ DCSSCmdType 0
        ("m_ERRCode", c_int),  # * Error code   */
    ]


# Define RT_IN structure
class RT_IN(Structure):  # noqa: N801 - Keep the SDK type name.
    _fields_ = [
        (
            "m_RtInSwitch",
            c_int,
        ),  # * Real-time input switch for user real-time data on/off: 0 - close rt_in; 1 - open rt_in *
        ("m_ImpType", c_int),  # Impedance type
        ("m_InFrameSerial", c_int),  # short input frame serial number 0 - 1000000 modulo
        ("m_FrameMissCnt", c_short),  # short frame miss count
        ("m_MaxFrameMissCnt", c_short),  # short max frame miss count before triggering backward
        ("m_SysCyc", c_int),  # 0 -  1000000
        ("m_SysCycMissCnt", c_short),  # short Real-time miss count
        ("m_MaxSysCycMissCnt", c_short),  # short max real-time miss count before triggering backward
        ("m_ToolKine", c_float * 6),  # Tool kinematics parameters 1
        ("m_ToolDyn", c_float * 10),  # Tool dynamics parameters 1
        ("m_Joint_CMD_Pos", c_float * 7),  # Joint position command
        ("m_Joint_Vel_Ratio", c_short),  # short Joint velocity limit percentage 2
        ("m_Joint_Acc_Ratio", c_short),  # short Joint acceleration limit percentage 2
        ("m_Joint_K", c_float * 7),  # Joint impedance stiffness K command 3
        ("m_Joint_D", c_float * 7),  # Joint impedance damping D command 4
        ("m_DragSpType", c_int),  # Nullspace type 5
        ("m_DragSpPara", c_float * 6),  # Nullspace parameter type 5
        ("m_Cart_KD_Type", c_int),  # Cartesian impedance type
        ("m_Cart_K", c_float * 6),  # Cartesian impedance stiffness K command 4
        ("m_Cart_D", c_float * 6),  # Cartesian impedance damping D command 4
        ("m_Cart_KN", c_float),  # 4
        ("m_Cart_DN", c_float),  # 4
        ("m_Force_FB_Type", c_int),  # Force control feedback source type
        ("m_Force_Type", c_int),  # Force control type 6
        ("m_Force_Dir", c_float * 6),  # Force control direction in 6D space 6
        ("m_Force_PIDUL", c_float * 7),  # Force control PID 6
        ("m_Force_AdjLmt", c_float),  # Maximum allowable adjustment range 6
        ("m_Force_Cmd", c_float),  # Force control command 8
        ("m_SET_Tags", c_ubyte * 16),  # Nullspace type 5
        ("m_Update_Tags", c_ubyte * 16),  # Nullspace type 5
        ("m_PvtID", c_ubyte),  # Set PVT ID
        ("m_PvtID_Update", c_ubyte),  # PVT ID update status
        ("m_Pvt_RunID", c_ubyte),  # 0: no pvt file; 1~99: user uploaded PVT
        ("m_Pvt_RunState", c_ubyte),  # 0: idle; 1: loading ; 2: running; 3: error
    ]


# Define RT_OUT structure
class RT_OUT(Structure):  # noqa: N801 - Keep the SDK type name.
    _fields_ = [
        ("m_OutFrameSerial", c_int),  # output frame serial number 0 - 1000000 modulo
        ("m_FB_Joint_Pos", c_float * 7),  # Joint position feedback
        ("m_FB_Joint_Vel", c_float * 7),  # Joint velocity feedback
        ("m_FB_Joint_PosE", c_float * 7),  # Joint position (external encoder)
        ("m_FB_Joint_Cmd", c_float * 7),  # Joint position command
        ("m_FB_Joint_CToq", c_float * 7),  # Joint command torque
        ("m_FB_Joint_SToq", c_float * 7),  # Joint actual torque
        ("m_FB_Joint_Them", c_float * 7),  # Joint temperature
        ("m_EST_Joint_Firc", c_float * 7),  # Joint friction estimate
        ("m_EST_Joint_Firc_Dot", c_float * 7),  # Joint force disturbance estimate derivative
        ("m_EST_Joint_Force", c_float * 7),  # Joint force disturbance estimate
        ("m_EST_Cart_FN", c_float * 6),  # End-effector Cartesian space force disturbance estimate
        ("m_TipDI", c_char),  # End-effector digital input
        ("m_LowSpdFlag", c_char),  # Low speed flag
        # ("m_pad", c_char * 2)  # Padding bytes
    ]


# Define DCSS structure
class DCSS(Structure):
    _fields_ = [
        ("m_State", StateCtr * 2),  # Status controller array
        ("m_In", RT_IN * 2),  # Input data array
        ("m_Out", RT_OUT * 2),  # Output data array
        ("m_ParaName", c_char * 30),  # Parameter name, related to robot configuration parameters
        ("m_ParaType", c_ubyte),  # 0: FX_INT32; 1: FX_DOUBLE; 2: FX_STRING
        ("m_ParaIns", c_ubyte),  # DCSSCfgOperationType
        ("m_ParaValueI", c_int),  # FX_INT32 value
        ("m_ParaValueF", c_float),  # FX_FLOAT value
        ("m_ParaCmdSerial", c_short),  # short from PC
        ("m_ParaRetSerial", c_short),  # short working: 0; finish: cmd serial; error cmd_serial + 100
    ]
