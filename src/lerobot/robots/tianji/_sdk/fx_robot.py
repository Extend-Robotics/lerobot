# Adapted from wuji-hand-teleop at 647801345a6a27dec5cbf56280ce63bb8b2f6a32.
# Copyright (c) 2025 Wuji Technology Co., Ltd. See LICENSE.

"""Read-only subset of the vendor Python wrapper for the Marvin SDK."""

import ctypes
from pathlib import Path

from .structure_data import DCSS


def structure2dict(dcss: DCSS) -> dict:
    result = {
        "para_name": ["Marvin_sub_data"],
        "states": [
            {
                "cur_state": dcss.m_State[0].m_CurState,
                "cmd_state": dcss.m_State[0].m_CmdState,
                "err_code": dcss.m_State[0].m_ERRCode,
            },
            {
                "cur_state": dcss.m_State[1].m_CurState,
                "cmd_state": dcss.m_State[1].m_CmdState,
                "err_code": dcss.m_State[1].m_ERRCode,
            },
        ],
    }
    # 3. Process real-time output array
    result["outputs"] = [
        {
            "frame_serial": rt_out.m_OutFrameSerial,
            "tip_di": rt_out.m_TipDI,
            "low_speed_flag": rt_out.m_LowSpdFlag,
            "fb_joint_pos": [round(rt_out.m_FB_Joint_Pos[j], 4) for j in range(7)],
            "fb_joint_vel": [round(rt_out.m_FB_Joint_Vel[j], 4) for j in range(7)],
            "fb_joint_posE": [round(rt_out.m_FB_Joint_PosE[j], 4) for j in range(7)],
            "fb_joint_cmd": [round(rt_out.m_FB_Joint_Cmd[j], 4) for j in range(7)],
            "fb_joint_cToq": [round(rt_out.m_FB_Joint_CToq[j], 4) for j in range(7)],
            "fb_joint_sToq": [round(rt_out.m_FB_Joint_SToq[j], 4) for j in range(7)],
            "fb_joint_them": [round(rt_out.m_FB_Joint_Them[j], 4) for j in range(7)],
            "est_joint_firc": [round(rt_out.m_EST_Joint_Firc[j], 4) for j in range(7)],
            "est_joint_firc_dot": [round(rt_out.m_EST_Joint_Firc_Dot[j], 4) for j in range(7)],
            "est_joint_force": [round(rt_out.m_EST_Joint_Force[j], 4) for j in range(7)],
            "est_cart_fn": [round(rt_out.m_EST_Cart_FN[j], 4) for j in range(6)],
        }
        for rt_out in dcss.m_Out
    ]

    # 4. Process real-time input array (RT_IN)
    result["inputs"] = [
        {
            "rt_in_switch": rt_in.m_RtInSwitch,
            "imp_type": rt_in.m_ImpType,
            "in_frame_serial": rt_in.m_InFrameSerial,
            "frame_miss_cnt": rt_in.m_FrameMissCnt,
            "max_frame_miss_cnt": rt_in.m_MaxFrameMissCnt,
            "sys_cyc": rt_in.m_SysCyc,
            "sys_cyc_miss_cnt": rt_in.m_SysCycMissCnt,
            "max_sys_cyc_miss_cnt": rt_in.m_MaxSysCycMissCnt,
            "tool_kine": [round(rt_in.m_ToolKine[j], 4) for j in range(6)],
            "tool_dyn": [round(rt_in.m_ToolDyn[j], 4) for j in range(10)],
            "joint_cmd_pos": [round(rt_in.m_Joint_CMD_Pos[j], 4) for j in range(7)],
            "joint_vel_ratio": rt_in.m_Joint_Vel_Ratio,
            "joint_acc_ratio": rt_in.m_Joint_Acc_Ratio,
            "joint_k": [round(rt_in.m_Joint_K[j], 4) for j in range(7)],
            "joint_d": [round(rt_in.m_Joint_D[j], 4) for j in range(7)],
            "drag_sp_type": rt_in.m_DragSpType,
            "drag_sp_para": [round(rt_in.m_DragSpPara[j], 4) for j in range(6)],
            "cart_kd_type": rt_in.m_Cart_KD_Type,
            "cart_k": [round(rt_in.m_Cart_K[j], 4) for j in range(6)],
            "cart_d": [round(rt_in.m_Cart_D[j], 4) for j in range(6)],
            "cart_kn": round(rt_in.m_Cart_KN, 4),
            "cart_dn": round(rt_in.m_Cart_DN, 4),
            "force_fb_type": rt_in.m_Force_FB_Type,
            "force_type": rt_in.m_Force_Type,
            "force_dir": [round(rt_in.m_Force_Dir[j], 4) for j in range(6)],
            "force_pidul": [round(rt_in.m_Force_PIDUL[j], 4) for j in range(7)],
            "force_adj_lmt": round(rt_in.m_Force_AdjLmt, 4),
            "force_cmd": round(rt_in.m_Force_Cmd, 4),
            "set_tags": list(rt_in.m_SET_Tags),
            "update_tags": list(rt_in.m_Update_Tags),
            "pvt_id": rt_in.m_PvtID,
            "pvt_id_update": rt_in.m_PvtID_Update,
            "pvt_run_id": rt_in.m_Pvt_RunID,
            "pvt_run_state": rt_in.m_Pvt_RunState,
        }
        for rt_in in dcss.m_In
    ]

    result["ParaName"] = [list(dcss.m_ParaName)]
    result["ParaType"] = [dcss.m_ParaType]
    result["ParaIns"] = [dcss.m_ParaIns]
    result["ParaValueI"] = [dcss.m_ParaValueI]
    result["ParaValueF"] = [dcss.m_ParaValueF]
    result["ParaCmdSerial"] = [dcss.m_ParaCmdSerial]
    result["ParaRetSerial"] = [dcss.m_ParaRetSerial]

    return result


class Marvin_Robot:  # noqa: N801 - Preserve the vendor wrapper name.
    """In-process connection to the Tianji cabinet."""

    def __init__(self):
        library_path = Path(__file__).resolve().parent / "lib" / "libMarvinSDK.so"
        if not library_path.is_file():
            raise FileNotFoundError(f"Install the Tianji Marvin SDK binary at {library_path}")
        self.robot = ctypes.CDLL(str(library_path))
        self.robot.OnLinkTo.argtypes = [ctypes.c_ubyte] * 4
        self.robot.OnLinkTo.restype = ctypes.c_int
        self.robot.OnGetBuf.argtypes = [ctypes.POINTER(DCSS)]
        self.robot.OnGetBuf.restype = ctypes.c_bool
        self.robot.OnClearSet.argtypes = []
        self.robot.OnClearSet.restype = ctypes.c_bool
        self.robot.OnSetSend.argtypes = []
        self.robot.OnSetSend.restype = ctypes.c_bool
        for arm in ("A", "B"):
            for name, argtypes in (
                ("OnSetJointCmdPos", [ctypes.POINTER(ctypes.c_double)]),
                ("OnSetJointLmt", [ctypes.c_int, ctypes.c_int]),
                ("OnSetTargetState", [ctypes.c_int]),
            ):
                function = getattr(self.robot, f"{name}_{arm}")
                function.argtypes = argtypes
                function.restype = ctypes.c_bool
        self.robot.OnRelease.argtypes = []
        self.robot.OnRelease.restype = ctypes.c_int

    def connect(self, robot_ip: str) -> int:
        octets = robot_ip.split(".")
        if len(octets) != 4 or any(not octet.isdecimal() or int(octet) > 255 for octet in octets):
            raise ValueError(f"Expected a Tianji IPv4 address, got {robot_ip!r}")
        return self.robot.OnLinkTo(*(int(octet) for octet in octets))

    def subscribe(self, dcss: DCSS) -> dict:
        self.robot.OnGetBuf(ctypes.byref(dcss))
        return structure2dict(dcss)

    def request_feedback(self) -> None:
        # Send only the SDK header, with zero instructions and no command data.
        if self.robot.OnClearSet():
            self.robot.OnSetSend()

    def send_joint_positions(
        self, left: list[float], right: list[float], *, speed_ratio: int | None = None
    ) -> None:
        self._check(self.robot.OnClearSet(), "clear the command buffer")
        for arm, joints in (("A", left), ("B", right)):
            if speed_ratio is not None:
                set_limits = getattr(self.robot, f"OnSetJointLmt_{arm}")
                self._check(set_limits(speed_ratio, speed_ratio), f"set arm {arm} speed and acceleration")
            set_position = getattr(self.robot, f"OnSetJointCmdPos_{arm}")
            self._check(set_position((ctypes.c_double * 7)(*joints)), f"stage arm {arm} positions")
        self._check(self.robot.OnSetSend(), "queue joint positions")

    def set_arm_state(self, state: int) -> None:
        self._check(self.robot.OnClearSet(), "clear the command buffer")
        self._check(self.robot.OnSetTargetState_A(state), f"set arm A state to {state}")
        self._check(self.robot.OnSetTargetState_B(state), f"set arm B state to {state}")
        self._check(self.robot.OnSetSend(), "queue arm state change")

    @staticmethod
    def _check(result: bool, operation: str) -> None:
        if not result:
            raise OSError(f"Marvin SDK could not {operation}")

    def release_robot(self) -> int:
        return self.robot.OnRelease()
