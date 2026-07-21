from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig


F1_JOINT_NAMES = [
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_yaw_joint",
  "left_wrist_pitch_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_yaw_joint",
  "right_wrist_pitch_joint",
]


class MotionLoader:
  """读取、插值并计算 F1 动作速度。"""

  def __init__(
    self,
    motion_file: str,
    input_fps: float,
    output_fps: float,
    device: torch.device | str,
    num_joints: int,
    line_range: tuple[int, int] | None = None,
  ):
    if input_fps <= 0 or output_fps <= 0:
      raise ValueError("input_fps 和 output_fps 必须大于 0。")

    self.motion_file = motion_file
    self.input_fps = float(input_fps)
    self.output_fps = float(output_fps)
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = torch.device(device)
    self.num_joints = num_joints
    self.line_range = line_range

    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """从 CSV 读取根节点位姿和关节角。"""
    motion_path = Path(self.motion_file).expanduser().resolve()
    if not motion_path.is_file():
      raise FileNotFoundError(f"找不到输入 CSV：{motion_path}")

    if self.line_range is None:
      motion_np = np.loadtxt(motion_path, delimiter=",")
    else:
      start_line, end_line = self.line_range
      if start_line < 1 or end_line < start_line:
        raise ValueError("line_range 必须满足 1 <= 起始行 <= 结束行。")
      motion_np = np.loadtxt(
        motion_path,
        delimiter=",",
        skiprows=start_line - 1,
        max_rows=end_line - start_line + 1,
      )

    # np.loadtxt 在只有一行数据时会返回一维数组，这里统一成二维。
    motion_np = np.atleast_2d(motion_np)
    expected_cols = 3 + 4 + self.num_joints
    if motion_np.shape[1] != expected_cols:
      raise ValueError(
        f"F1 CSV 列数不正确：当前为 {motion_np.shape[1]} 列，"
        f"应为 {expected_cols} 列，即 root_pos(3) + root_quat_xyzw(4) "
        f"+ joint_pos({self.num_joints})。"
      )
    if motion_np.shape[0] < 3:
      raise ValueError("至少需要 3 帧动作数据才能计算线速度和角速度。")
    if not np.isfinite(motion_np).all():
      raise ValueError("CSV 中包含 NaN 或 Inf，请先清理动作数据。")

    motion = torch.from_numpy(motion_np).to(
      device=self.device,
      dtype=torch.float32,
    )

    self.motion_base_poss_input = motion[:, 0:3]

    # 输入 CSV 使用 xyzw，MJLab 内部使用 wxyz。
    self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]
    self.motion_base_rots_input = torch.nn.functional.normalize(
      self.motion_base_rots_input,
      dim=-1,
    )
    self._fix_quaternion_sign()

    self.motion_dof_poss_input = motion[:, 7:]
    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

    print("\n===== F1 CSV 信息 =====")
    print(f"输入文件：{motion_path}")
    print(f"输入帧数：{self.input_frames}")
    print(f"输入频率：{self.input_fps:.3f} Hz")
    print(f"动作时长：{self.duration:.3f} s")
    print(f"关节数量：{self.motion_dof_poss_input.shape[1]}")

  def _fix_quaternion_sign(self):
    """消除 q 和 -q 等价表示导致的插值跳变。"""
    for i in range(1, self.motion_base_rots_input.shape[0]):
      previous = self.motion_base_rots_input[i - 1]
      current = self.motion_base_rots_input[i]
      if torch.dot(previous, current) < 0:
        self.motion_base_rots_input[i] = -current

  def _interpolate_motion(self):
    """将输入动作重采样到目标频率。"""
    self.output_frames = int(np.floor(self.duration / self.output_dt)) + 1
    times = (
      torch.arange(
        self.output_frames,
        device=self.device,
        dtype=torch.float32,
      )
      * self.output_dt
    )
    times = torch.clamp(times, max=self.duration)

    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )

    print("\n===== 插值结果 =====")
    print(f"输出帧数：{self.output_frames}")
    print(f"输出频率：{self.output_fps:.3f} Hz")

  @staticmethod
  def _lerp(
    a: torch.Tensor,
    b: torch.Tensor,
    blend: torch.Tensor,
  ) -> torch.Tensor:
    """线性插值。"""
    return a * (1.0 - blend) + b * blend

  @staticmethod
  def _slerp(
    a: torch.Tensor,
    b: torch.Tensor,
    blend: torch.Tensor,
  ) -> torch.Tensor:
    """对四元数执行球面线性插值。"""
    result = torch.zeros_like(a)
    for i in range(a.shape[0]):
      result[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return torch.nn.functional.normalize(result, dim=-1)

  def _compute_frame_blend(
    self,
    times: torch.Tensor,
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """计算每个输出时刻对应的输入帧和插值比例。"""
    frame_position = times / self.input_dt
    index_0 = torch.floor(frame_position).long()
    index_0 = torch.clamp(index_0, min=0, max=self.input_frames - 1)
    index_1 = torch.clamp(index_0 + 1, max=self.input_frames - 1)
    blend = frame_position - index_0.to(frame_position.dtype)
    blend = torch.where(index_0 == index_1, torch.zeros_like(blend), blend)
    return index_0, index_1, blend

  def _compute_velocities(self):
    """根据插值后的位姿计算根节点与关节速度。"""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss,
      spacing=self.output_dt,
      dim=0,
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss,
      spacing=self.output_dt,
      dim=0,
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots,
      self.output_dt,
    )

  @staticmethod
  def _so3_derivative(
    rotations: torch.Tensor,
    dt: float,
  ) -> torch.Tensor:
    """用中心差分计算世界坐标系下的角速度。"""
    q_prev = rotations[:-2]
    q_next = rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))
    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)

    # 首尾帧沿用邻近帧结果，使输出长度与动作帧数一致。
    return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    """返回下一帧状态，并在动作结束时给出重置标志。"""
    idx = self.current_idx
    state = (
      self.motion_base_poss[idx : idx + 1],
      self.motion_base_rots[idx : idx + 1],
      self.motion_base_lin_vels[idx : idx + 1],
      self.motion_base_ang_vels[idx : idx + 1],
      self.motion_dof_poss[idx : idx + 1],
      self.motion_dof_vels[idx : idx + 1],
    )

    self.current_idx += 1
    reset_flag = self.current_idx >= self.output_frames
    if reset_flag:
      self.current_idx = 0
    return state, reset_flag


def load_f1_scene_cfg(
  env_cfg_module: str,
  env_cfg_func: str,
):
  """动态加载 F1 跟踪环境配置，避免把项目路径写死在脚本中。"""
  try:
    module = importlib.import_module(env_cfg_module)
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      f"无法导入 F1 环境配置模块：{env_cfg_module}\n"
      "请把 --env-cfg-module 改成你项目中 env_cfgs.py 的真实模块路径。"
    ) from exc

  if not hasattr(module, env_cfg_func):
    available = [
      name
      for name in dir(module)
      if name.endswith("tracking_env_cfg") and not name.startswith("_")
    ]
    raise AttributeError(
      f"模块 {env_cfg_module} 中不存在函数 {env_cfg_func}。\n"
      f"当前可疑的配置函数：{available}"
    )

  env_cfg_factory = getattr(module, env_cfg_func)
  env_cfg = env_cfg_factory()
  return env_cfg.scene


def validate_robot_joints(
  robot: Entity,
  joint_names: list[str],
):
  """确认 F1 环境中的关节名称和 CSV 顺序可以完整匹配。"""
  robot_joint_indexes = robot.find_joints(
    joint_names,
    preserve_order=True,
  )[0]

  if len(robot_joint_indexes) != len(joint_names):
    raise RuntimeError(
      f"只匹配到 {len(robot_joint_indexes)}/{len(joint_names)} 个 F1 关节。\n"
      f"环境中的关节：{robot.joint_names}\n"
      f"脚本要求的关节：{joint_names}"
    )

  matched_names = [robot.joint_names[i] for i in robot_joint_indexes]
  if matched_names != joint_names:
    raise RuntimeError(
      f"F1 关节顺序匹配异常。\n期望顺序：{joint_names}\n实际顺序：{matched_names}"
    )

  print("\n===== F1 关节检查通过 =====")
  for index, name in enumerate(matched_names):
    print(f"{index:02d}: {name}")
  return robot_joint_indexes


def save_motion_log(
  log: dict[str, Any],
  output_file: str,
):
  """堆叠每帧数据并保存为 NPZ。"""
  array_keys = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  )
  for key in array_keys:
    log[key] = np.stack(log[key], axis=0)

  output_path = Path(output_file).expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(output_path, **log)

  print("\n===== NPZ 保存完成 =====")
  print(f"保存路径：{output_path}")
  for key in array_keys:
    print(f"{key}: {log[key].shape}")
  return output_path


def upload_to_wandb(
  output_path: Path,
  output_name: str,
  project_name: str,
  video_path: Path | None = None,
):
  """把 NPZ 和可选视频上传到 W&B。"""
  import wandb

  registry_name = "motions"
  run = wandb.init(project=project_name, name=output_name)
  print(f"[信息] 正在上传动作：{output_name}")

  logged_artifact = run.log_artifact(
    artifact_or_path=str(output_path),
    name=output_name,
    type=registry_name,
  )
  run.link_artifact(
    artifact=logged_artifact,
    target_path=f"wandb-registry-{registry_name}/{output_name}",
  )

  if video_path is not None:
    wandb.log(
      {
        "motion_video": wandb.Video(
          str(video_path),
          format="mp4",
        )
      }
    )

  wandb.finish()
  print(f"[信息] 已上传到 W&B motions/{output_name}")


def run_sim(
  sim: Simulation,
  scene: Scene,
  input_file: str,
  input_fps: float,
  output_fps: float,
  output_name: str,
  output_file: str,
  render: bool,
  line_range: tuple[int, int] | None,
  upload_wandb_enabled: bool,
  wandb_project: str,
  renderer: OffscreenRenderer | None = None,
):
  """在 F1 模型上回放 CSV，并记录完整刚体运动学数据。"""
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    num_joints=len(F1_JOINT_NAMES),
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = validate_robot_joints(robot, F1_JOINT_NAMES)

  log: dict[str, Any] = {
    "fps": np.asarray([output_fps], dtype=np.float32),
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }

  frames: list[np.ndarray] = []
  scene.reset()

  print(f"\n开始处理，共 {motion.output_frames} 帧。")
  if render:
    print("已启用离屏渲染，将生成动作视频。")

  progress = tqdm(
    total=motion.output_frames,
    desc="处理动作帧",
    unit="帧",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  while True:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    # 写入根节点状态。
    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, 0:2] += scene.env_origins[:, 0:2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:13] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    # 只覆盖 F1 的 28 个目标关节，其余固定关节保持默认状态。
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # 前向运动学更新后，读取全部刚体状态。
    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
    log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
    log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

    # 根刚体的速度应与写入的根节点速度一致。
    torch.testing.assert_close(
      robot.data.body_link_lin_vel_w[0, 0],
      motion_base_lin_vel[0],
      rtol=1e-4,
      atol=1e-5,
    )
    torch.testing.assert_close(
      robot.data.body_link_ang_vel_w[0, 0],
      motion_base_ang_vel[0],
      rtol=1e-4,
      atol=1e-5,
    )

    frame_count += 1
    progress.update(1)
    if frame_count % 100 == 0:
      elapsed_time = frame_count / output_fps
      progress.set_description(f"处理动作帧 t={elapsed_time:.1f}s")

    if reset_flag:
      break

  progress.close()
  output_path = save_motion_log(log, output_file)

  video_path = None
  if render:
    import mediapy as media

    video_path = output_path.with_suffix(".mp4")
    print(f"正在生成视频：{video_path}")
    media.write_video(str(video_path), frames, fps=output_fps)

  if upload_wandb_enabled:
    upload_to_wandb(
      output_path=output_path,
      output_name=output_name,
      project_name=wandb_project,
      video_path=video_path,
    )


def main(
  input_file: str,
  output_name: str,
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  output_file: str = "/tmp/f1_motion.npz",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
  upload_wandb: bool = True,
  wandb_project: str = "csv_to_npz",
  env_cfg_module: str = "mjlab.tasks.tracking.config.f1.env_cfgs",
  env_cfg_func: str = "f1_flat_tracking_env_cfg",
):
  """将 F1 动作 CSV 转换为 MJLab 跟踪任务使用的 NPZ。

  参数：
    input_file: 输入 CSV 文件路径。
    output_name: W&B 中的动作名称。
    input_fps: CSV 原始帧率。
    output_fps: NPZ 目标帧率。
    device: 运行设备，例如 cuda:0 或 cpu。
    output_file: 本地 NPZ 保存路径。
    render: 是否渲染并保存 MP4 视频。
    line_range: 只处理指定行，行号从 1 开始，包含首尾两行。
    upload_wandb: 是否上传到 W&B。
    wandb_project: W&B 项目名。
    env_cfg_module: F1 env_cfgs.py 的 Python 模块路径。
    env_cfg_func: 创建 F1 跟踪环境配置的函数名。
  """
  if device.startswith("cuda") and not torch.cuda.is_available():
    print("[警告] CUDA 不可用，自动切换到 CPU。")
    device = "cpu"

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene_cfg = load_f1_scene_cfg(
    env_cfg_module=env_cfg_module,
    env_cfg_func=env_cfg_func,
  )
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()

  sim = Simulation(
    num_envs=1,
    cfg=sim_cfg,
    model=model,
    device=device,
  )
  scene.initialize(sim.mj_model, sim.model, sim.data)

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot",
      distance=2.2,
      elevation=-5.0,
      azimuth=20.0,
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  run_sim(
    sim=sim,
    scene=scene,
    input_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    output_name=output_name,
    output_file=output_file,
    render=render,
    line_range=line_range,
    upload_wandb_enabled=upload_wandb,
    wandb_project=wandb_project,
    renderer=renderer,
  )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
