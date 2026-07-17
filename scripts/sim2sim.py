"""在 MuJoCo 中运行导出的 F1 坐立追踪策略。

由 ``MotionTrackingOnPolicyRunner`` 导出的追踪 ONNX 模型有两个输入：
``obs`` 和 ``time_step``。除策略动作外，它还返回 ``time_step`` 对应的参考
运动帧。本脚本利用这些内嵌的参考帧重建 actor 观测，并在独立的 MuJoCo
仿真中执行位置目标。

当前 Actor 的单帧观测顺序为：
``command``、``projected_gravity``、``base_ang_vel``、``joint_pos``、
``joint_vel``、``actions``。每项保留 5 帧历史，最终观测维度为 730。

示例：
  uv run scripts/sim2sim.py \
    logs/rsl_rl/f1_sit_to_stand_tracking/2026-07-16_15-19-54/2026-07-16_15-19-54.onnx
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import onnx
import onnxruntime as ort

from mjlab.asset_zoo.robots import get_chair_cfg, get_f1_robot_cfg
from mjlab.scene import Scene, SceneCfg
from mjlab.terrains import TerrainEntityCfg

DEFAULT_ONNX = Path(
  "logs/rsl_rl/f1_sit_to_stand_tracking/20000/2026-07-16_20-00-57.onnx"
)

ROBOT_PREFIX = "robot/"

HISTORY_LENGTH = 5
ACTOR_OBSERVATION_NAMES = (
  "command",
  "projected_gravity",
  "base_ang_vel",
  "joint_pos",
  "joint_vel",
  "actions",
)
# 单帧维度：56 + 3 + 3 + 28 + 28 + 28 = 146。
ACTOR_OBSERVATION_DIM = 146 * HISTORY_LENGTH
GRAVITY_VECTOR_W = np.array([0.0, 0.0, -1.0], dtype=np.float32)


def _csv_str(value: str) -> list[str]:
  # 解析逗号分隔的字符串列表
  return [item for item in value.split(",") if item]


def _csv_float(value: str) -> np.ndarray:
  # 解析逗号分隔的浮点数数组
  return np.array([float(item) for item in value.split(",")], dtype=np.float32)


def _read_metadata(onnx_path: Path) -> dict[str, str]:
  # 读取 ONNX 模型的元数据
  model = onnx.load(str(onnx_path))
  return {prop.key: prop.value for prop in model.metadata_props}


def _require_metadata(metadata: dict[str, str], keys: Iterable[str]) -> None:
  # 检查必要的元数据键是否存在
  missing = [key for key in keys if key not in metadata]
  if missing:
    raise KeyError(f"ONNX 元数据缺少必要字段: {missing}")


def _joint_name(model: mujoco.MjModel, joint_id: int) -> str:
  # 去掉关节名称中的机器人前缀
  name = model.joint(joint_id).name
  if name.startswith(ROBOT_PREFIX):
    return name[len(ROBOT_PREFIX) :]
  return name


def _sensor_slice(model: mujoco.MjModel, name: str) -> slice:
  # 返回传感器数据在 sensordata 数组中的切片范围
  sensor_name = f"{ROBOT_PREFIX}{name}"
  sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
  if sensor_id < 0:
    raise KeyError(f"传感器未找到: {sensor_name}")
  start = int(model.sensor_adr[sensor_id])
  stop = start + int(model.sensor_dim[sensor_id])
  return slice(start, stop)


def _quat_conj(quat: np.ndarray) -> np.ndarray:
  result = quat.copy()
  result[..., 1:] *= -1.0
  return result


def _quat_mul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
  lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
  rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
  return np.stack(
    (
      lw * rw - lx * rx - ly * ry - lz * rz,
      lw * rx + lx * rw + ly * rz - lz * ry,
      lw * ry - lx * rz + ly * rw + lz * rx,
      lw * rz + lx * ry - ly * rx + lz * rw,
    ),
    axis=-1,
  )


def _quat_apply_inverse(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  zeros = np.zeros((*vec.shape[:-1], 1), dtype=vec.dtype)
  vec_quat = np.concatenate((zeros, vec), axis=-1)
  return _quat_mul(_quat_mul(_quat_conj(quat), vec_quat), quat)[..., 1:]


def _relative_transform(
  parent_pos: np.ndarray,
  parent_quat: np.ndarray,
  child_pos: np.ndarray,
  child_quat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  rel_pos = _quat_apply_inverse(parent_quat, child_pos - parent_pos)
  rel_quat = _quat_mul(_quat_conj(parent_quat), child_quat)
  return rel_pos, rel_quat


def _matrix_from_quat(quat: np.ndarray) -> np.ndarray:
  quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1e-8)
  w, x, y, z = np.moveaxis(quat, -1, 0)
  ww, xx, yy, zz = w * w, x * x, y * y, z * z
  wx, wy, wz = w * x, w * y, w * z
  xy, xz, yz = x * y, x * z, y * z
  row0 = np.stack((ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)), axis=-1)
  row1 = np.stack((2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)), axis=-1)
  row2 = np.stack((2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz), axis=-1)
  return np.stack((row0, row1, row2), axis=-2)


def _first_two_rotation_columns(quat: np.ndarray) -> np.ndarray:
  return _matrix_from_quat(quat)[..., :2].reshape(-1).astype(np.float32)


class F1Sim2Sim:
  def __init__(self, onnx_path: Path) -> None:
    self.onnx_path = onnx_path
    self.metadata = _read_metadata(onnx_path)
    _require_metadata(
      self.metadata,
      (
        "joint_names",
        "default_joint_pos",
        "action_scale",
        "anchor_body_name",
        "body_names",
        "observation_names",
      ),
    )
    self.joint_names = _csv_str(self.metadata["joint_names"])
    self.body_names = _csv_str(self.metadata["body_names"])
    self.default_joint_pos = _csv_float(self.metadata["default_joint_pos"])
    self.action_scale = _csv_float(self.metadata["action_scale"])
    self.observation_names = _csv_str(self.metadata["observation_names"])
    if tuple(self.observation_names) != ACTOR_OBSERVATION_NAMES:
      raise ValueError(
        "ONNX Actor 观测顺序与当前 Sim2Sim 脚本不一致。\n"
        f"脚本要求: {list(ACTOR_OBSERVATION_NAMES)}\n"
        f"ONNX 实际: {self.observation_names}"
      )

    self.history_length = HISTORY_LENGTH
    self.last_action = np.zeros(len(self.joint_names), dtype=np.float32)
    self.term_history: dict[str, deque[np.ndarray]] = {}

    providers = ["CPUExecutionProvider"]
    self.session = ort.InferenceSession(str(onnx_path), providers=providers)
    self.input_names = [inp.name for inp in self.session.get_inputs()]
    if self.input_names != ["obs", "time_step"]:
      raise ValueError(f"ONNX 输入名称不符合预期: {self.input_names}")

    obs_input = self.session.get_inputs()[0]
    obs_dim = obs_input.shape[-1]
    if isinstance(obs_dim, int) and obs_dim != ACTOR_OBSERVATION_DIM:
      raise ValueError(
        f"ONNX obs 维度为 {obs_dim}，"
        f"当前脚本要求 {ACTOR_OBSERVATION_DIM}。"
      )

    self.scene = Scene(
      SceneCfg(
        num_envs=1,
        terrain=TerrainEntityCfg(terrain_type="plane"),
        entities={"robot": get_f1_robot_cfg(), "chair": get_chair_cfg()},
      ),
      device="cpu",
    )
    self.model = self.scene.compile()
    self.data = mujoco.MjData(self.model)
    mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
    self.model.opt.timestep = 0.005

    self._build_indices()
    self.imu_ang_vel_slice = _sensor_slice(self.model, "imu_ang_vel")

    zero_obs = np.zeros((1, ACTOR_OBSERVATION_DIM), dtype=np.float32)
    self.reference0 = self._run_onnx(zero_obs, 0)
    self._reset_to_reference(self.reference0)
    mujoco.mj_forward(self.model, self.data)

  def _build_indices(self) -> None:
    joint_ids = []
    joint_qadr = []
    joint_dadr = []
    for name in self.joint_names:
      joint_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{ROBOT_PREFIX}{name}"
      )
      if joint_id < 0:
        raise KeyError(f"模型中未找到关节: {name}")
      joint_ids.append(joint_id)
      joint_qadr.append(int(self.model.jnt_qposadr[joint_id]))
      joint_dadr.append(int(self.model.jnt_dofadr[joint_id]))
    self.joint_ids = np.array(joint_ids, dtype=np.int32)
    self.joint_qadr = np.array(joint_qadr, dtype=np.int32)
    self.joint_dadr = np.array(joint_dadr, dtype=np.int32)

    body_ids = []
    for name in self.body_names:
      body_id = mujoco.mj_name2id(
        self.model, mujoco.mjtObj.mjOBJ_BODY, f"{ROBOT_PREFIX}{name}"
      )
      if body_id < 0:
        raise KeyError(f"模型中未找到刚体: {name}")
      body_ids.append(body_id)
    self.body_ids = np.array(body_ids, dtype=np.int32)
    self.anchor_body_index = self.body_names.index(self.metadata["anchor_body_name"])
    self.anchor_body_id = self.body_ids[self.anchor_body_index]
    self.root_joint_id = mujoco.mj_name2id(
      self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{ROBOT_PREFIX}floating_base_joint"
    )
    if self.root_joint_id < 0:
      raise KeyError("模型中未找到 floating_base_joint。")
    self.root_qadr = int(self.model.jnt_qposadr[self.root_joint_id])
    self.root_dadr = int(self.model.jnt_dofadr[self.root_joint_id])
    self.root_body_id = int(self.model.jnt_bodyid[self.root_joint_id])

    ctrl_for_policy_joint = np.empty(len(self.joint_names), dtype=np.int32)
    joint_name_to_policy_index = {
      name: index for index, name in enumerate(self.joint_names)
    }
    for ctrl_id in range(self.model.nu):
      target_joint_id = int(self.model.actuator_trnid[ctrl_id, 0])
      target_name = _joint_name(self.model, target_joint_id)
      policy_index = joint_name_to_policy_index[target_name]
      ctrl_for_policy_joint[policy_index] = ctrl_id
    self.ctrl_for_policy_joint = ctrl_for_policy_joint

    actuator_targets = [
      _joint_name(self.model, int(self.model.actuator_trnid[i, 0]))
      for i in range(self.model.nu)
    ]
    missing = set(self.joint_names) - set(actuator_targets)
    if missing:
      raise RuntimeError(f"策略关节缺少执行器: {sorted(missing)}")

  def _run_onnx(self, obs: np.ndarray, time_step: int) -> dict[str, np.ndarray]:
    outputs = self.session.run(
      None,
      {
        "obs": obs.astype(np.float32, copy=False),
        "time_step": np.array([[time_step]], dtype=np.float32),
      },
    )
    output_names = [out.name for out in self.session.get_outputs()]
    return dict(zip(output_names, outputs, strict=True))

  def _reset_to_reference(self, reference: dict[str, np.ndarray]) -> None:
    body_pos = reference["body_pos_w"][0]
    body_quat = reference["body_quat_w"][0]
    body_lin_vel = reference["body_lin_vel_w"][0]
    body_ang_vel = reference["body_ang_vel_w"][0]
    self.data.qpos[self.root_qadr : self.root_qadr + 3] = body_pos[0]
    self.data.qpos[self.root_qadr + 3 : self.root_qadr + 7] = body_quat[0]
    self.data.qvel[self.root_dadr : self.root_dadr + 3] = body_lin_vel[0]
    self.data.qvel[self.root_dadr + 3 : self.root_dadr + 6] = body_ang_vel[0]
    self.data.qpos[self.joint_qadr] = reference["joint_pos"][0]
    self.data.qvel[self.joint_dadr] = reference["joint_vel"][0]
    self.data.ctrl[self.ctrl_for_policy_joint] = self.default_joint_pos

  def _term_with_history(self, name: str, value: np.ndarray) -> np.ndarray:
    value = value.astype(np.float32, copy=False).reshape(-1)
    if name not in self.term_history:
      self.term_history[name] = deque(
        (value.copy() for _ in range(self.history_length)),
        maxlen=self.history_length,
      )
    else:
      self.term_history[name].append(value.copy())
    return np.concatenate(tuple(self.term_history[name]), axis=0)

  def _make_obs(self, reference: dict[str, np.ndarray]) -> np.ndarray:
    # 参考命令由当前参考帧的 28 维关节角和 28 维关节速度组成。
    motion_joint_pos = reference["joint_pos"][0].astype(np.float32)
    motion_joint_vel = reference["joint_vel"][0].astype(np.float32)
    command = np.concatenate((motion_joint_pos, motion_joint_vel), axis=0)

    # 将世界坐标系的单位重力方向投影到 pelvis 局部坐标系。
    root_quat_w = self.data.xquat[self.root_body_id].astype(np.float32)
    projected_gravity = _quat_apply_inverse(
      root_quat_w,
      GRAVITY_VECTOR_W,
    ).astype(np.float32)

    joint_pos = self.data.qpos[self.joint_qadr].astype(np.float32)
    joint_vel = self.data.qvel[self.joint_dadr].astype(np.float32)
    terms = {
      "command": command,
      "projected_gravity": projected_gravity,
      "base_ang_vel": self.data.sensordata[
        self.imu_ang_vel_slice
      ].astype(np.float32),
      "joint_pos": joint_pos - self.default_joint_pos,
      "joint_vel": joint_vel,
      "actions": self.last_action,
    }

    # 严格按照 ONNX metadata 中的观测顺序拼接，每项内部为旧帧到新帧。
    obs = np.concatenate(
      [
        self._term_with_history(name, terms[name])
        for name in self.observation_names
      ],
      axis=0,
    )
    if obs.shape != (ACTOR_OBSERVATION_DIM,):
      term_shapes = {name: terms[name].shape for name in self.observation_names}
      raise RuntimeError(
        f"期望观测形状为 ({ACTOR_OBSERVATION_DIM},)，"
        f"实际为 {obs.shape}，单帧观测形状为 {term_shapes}"
      )
    return obs[None, :]

  def step(
    self, control_step: int, decimation: int
  ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    reference = self._run_onnx(np.zeros((1, ACTOR_OBSERVATION_DIM), dtype=np.float32), control_step)
    obs = self._make_obs(reference)
    policy_out = self._run_onnx(obs, control_step)
    action = policy_out["actions"][0].astype(np.float32)
    self.last_action = action.copy()

    targets = self.default_joint_pos + action * self.action_scale
    self.data.ctrl[self.ctrl_for_policy_joint] = targets

    for _ in range(decimation):
      mujoco.mj_step(self.model, self.data)
    return action, policy_out

  def run(self, duration: float, decimation: int, realtime: bool) -> None:
    control_dt = self.model.opt.timestep * decimation
    max_steps = int(duration / control_dt)
    with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
      start = time.time()
      for control_step in range(max_steps):
        step_start = time.time()
        self.step(control_step, decimation)
        viewer.sync()
        if not viewer.is_running():
          break
        if realtime:
          elapsed = time.time() - step_start
          time.sleep(max(0.0, control_dt - elapsed))
      sim_time = max_steps * control_dt
      wall_time = time.time() - start
      print(f"sim_time={sim_time:.3f}s wall_time={wall_time:.3f}s")

  def run_headless(self, duration: float, decimation: int) -> None:
    control_dt = self.model.opt.timestep * decimation
    max_steps = int(duration / control_dt)
    start = time.time()
    for control_step in range(max_steps):
      self.step(control_step, decimation)
    sim_time = max_steps * control_dt
    wall_time = time.time() - start
    root_height = float(self.data.qpos[self.root_qadr + 2])
    print(
      f"sim_time={sim_time:.3f}s wall_time={wall_time:.3f}s "
      f"root_height={root_height:.3f}m"
    )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "onnx",
    nargs="?",
    type=Path,
    default=DEFAULT_ONNX,
    help="导出的追踪 ONNX 策略文件路径。",
  )
  parser.add_argument("--duration", type=float, default=5.0)
  parser.add_argument("--decimation", type=int, default=4)
  parser.add_argument(
    "--no-realtime",
    action="store_true",
    help="以最快速度运行，不与实际时钟同步。",
  )
  parser.add_argument(
    "--headless",
    action="store_true",
    help="不打开 MuJoCo 可视化窗口，直接运行。",
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  sim = F1Sim2Sim(args.onnx)
  if args.headless:
    sim.run_headless(duration=args.duration, decimation=args.decimation)
  else:
    sim.run(
      duration=args.duration,
      decimation=args.decimation,
      realtime=not args.no_realtime,
    )


if __name__ == "__main__":
  main()