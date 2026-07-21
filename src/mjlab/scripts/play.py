import json
from mjlab.envs.mdp.actions import JointPositionAction
import os
import sys
import time as _time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts._cli import maybe_print_top_level_help
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago


def _parse_wandb_dt(value: str | datetime) -> datetime:
  """解析W&B日期时间字符串或传递datetime对象。"""
  if isinstance(value, str):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
  return value


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False
  log_root: str = "logs/rsl_rl"
  action_trace_file: str | None = None
  # 当num_envs > 1时要记录的环境索引
  action_trace_env_idx: int = 0
  # 要记录的最大策略步数
  action_trace_max_steps: int = 2000
  # 演示脚本使用的内部标志
  _demo_mode: tyro.conf.Suppress[bool] = False


class ActionTracePolicy:
  """包装策略并记录关节位置目标值用于离线安全检查。"""

  def __init__(
    self,
    policy,
    env: RslRlVecEnvWrapper,
    output_file: str,
    env_idx: int = 0,
    max_steps: int = 2000,
  ):
    self.policy = policy
    self.env = env
    self.output_file = Path(output_file)
    self.env_idx = env_idx
    self.max_steps = max_steps
    self.step = 0

    joint_action = env.unwrapped.action_manager.get_term("joint_pos")
    assert isinstance(joint_action, JointPositionAction)
    self.joint_action = joint_action

    self.dt = env.unwrapped.step_dt
    self.joint_names = list(joint_action.target_names)

    self.prev_target_q: torch.Tensor | None = None
    self.prev_dq_target: torch.Tensor | None = None
    # 存储记录的轨迹数据列表
    self.records: list[dict] = []

  def __call__(self, obs) -> torch.Tensor:
    """调用策略获取动作并记录。"""
    actions = self.policy(obs)

    if self.step < self.max_steps:
      self._record(actions)

    self.step += 1
    return actions

  def _record(self, actions: torch.Tensor) -> None:
    """记录关节位置、速度和加速度目标值。"""
    # 匹配RslRlVecEnvWrapper传递给env.step()的动作
    if self.env.clip_actions is not None:
      actions_for_log = torch.clamp(
        actions,
        -self.env.clip_actions,
        self.env.clip_actions,
      )
    else:
      actions_for_log = actions

    scale = self.joint_action.scale
    offset = self.joint_action.offset

    # 应用缩放和偏移后的关节位置命令
    target_q = actions_for_log * scale + offset

    # 更接近JointPositionAction.apply_actions()写入仿真的值
    robot = self.env.unwrapped.scene[self.joint_action.cfg.entity_name]
    encoder_bias = robot.data.encoder_bias[:, self.joint_action.target_ids]
    sim_target_q = target_q - encoder_bias

    # 计算关节速度和加速度目标
    if self.prev_target_q is None:
      dq_target = torch.zeros_like(target_q)
      ddq_target = torch.zeros_like(target_q)
    else:
      dq_target = (target_q - self.prev_target_q) / self.dt
      assert self.prev_dq_target is not None
      ddq_target = (dq_target - self.prev_dq_target) / self.dt

    # 存储单个环境的记录
    env_i = self.env_idx
    self.records.append(
      {
        "step": self.step,
        "target_q": target_q[env_i].detach().cpu().tolist(),
        "sim_target_q": sim_target_q[env_i].detach().cpu().tolist(),
        "dq_target": dq_target[env_i].detach().cpu().tolist(),
        "ddq_target": ddq_target[env_i].detach().cpu().tolist(),
        "max_abs_dq_target": dq_target[env_i].abs().max().item(),
        "max_abs_ddq_target": ddq_target[env_i].abs().max().item(),
        "raw_action": actions_for_log[env_i].detach().cpu().tolist(),
      }
    )

    self.prev_target_q = target_q.detach().clone()
    self.prev_dq_target = dq_target.detach().clone()

  def save(self) -> None:
    """将记录的轨迹数据保存到JSON文件。"""
    if not self.records:
      return

    self.output_file.parent.mkdir(parents=True, exist_ok=True)

    target_q = torch.tensor([r["target_q"] for r in self.records])
    dq_target = torch.tensor([r["dq_target"] for r in self.records])
    ddq_target = torch.tensor([r["ddq_target"] for r in self.records])

    # 计算每个关节的统计摘要
    per_joint_summary = []
    for i, name in enumerate(self.joint_names):
      per_joint_summary.append(
        {
          "joint": name,
          "target_q_min": target_q[:, i].min().item(),
          "target_q_max": target_q[:, i].max().item(),
          "max_abs_dq_target": dq_target[:, i].abs().max().item(),
          "max_abs_ddq_target": ddq_target[:, i].abs().max().item(),
        }
      )

    payload = {
      "dt": self.dt,
      "control_hz": 1.0 / self.dt,
      "env_idx": self.env_idx,
      "joint_names": self.joint_names,
      "num_steps": len(self.records),
      "summary": {
        "max_abs_dq_target": dq_target.abs().max().item(),
        "max_abs_ddq_target": ddq_target.abs().max().item(),
      },
      "per_joint_summary": per_joint_summary,
      "records": self.records,
    }

    with open(self.output_file, "w") as f:
      json.dump(payload, f, indent=2)

    print(f"[INFO]: Saved action trace to {self.output_file}")


def run_play(task_id: str, cfg: PlayConfig):
  """主函数:加载环境、代理和查看器，运行交互式演示。"""
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # 如果请求则禁用终止条件(用于查看运动)
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # 检查是否是跟踪任务(通过检查是否有运动命令)
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  # 演示模式:使用均匀采样以便在num_envs > 1时看到更多多样性
  if is_tracking_task and cfg._demo_mode:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  # 处理跟踪任务的运动文件加载
  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # 首先检查本地运动文件(适用于虚拟和已训练模式)
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      # 虚拟模式:从WandB注册表下载运动文件
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
      # 检查注册表名称是否包含版本别名,如果没有则添加":latest"
      registry_name = cfg.registry_name
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      # 已训练模式:从指定位置加载运动文件或从WandB下载
      if cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        # 从WandB运行中获取运动工件
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")

  # 初始化日志目录和检查点路径
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path(cfg.log_root) / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      # 使用本地检查点文件
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      # 从WandB下载检查点
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path), cfg.wandb_checkpoint_name
      )
      # 从路径中提取运行ID和检查点名称用于显示
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  # 配置环境参数
  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  # 创建环境
  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  # 如果需要则添加视频录制包装器
  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  # 包装环境以支持动作剪裁
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    # 虚拟模式:创建零或随机策略
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":
      # 零策略:始终输出零动作
      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:
      # 随机策略:输出[-1, 1]范围内的随机动作
      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    # 已训练模式:从检查点加载训练好的策略
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # 如果需要则包装策略以记录动作轨迹
  trace_policy: ActionTracePolicy | None = None
  if cfg.action_trace_file is not None:
    trace_policy = ActionTracePolicy(
      policy=policy,
      env=env,
      output_file=cfg.action_trace_file,
      env_idx=cfg.action_trace_env_idx,
      max_steps=cfg.action_trace_max_steps,
    )
    policy = trace_policy

  # 为查看器中的检查点热交换构建检查点管理器
  ckpt_manager: CheckpointManager | None = None
  if TRAINED_MODE and resume_path is not None:
    _ckpt_runner = runner  # pyright: ignore[reportPossiblyUnboundVariable]

    def _reload_policy(path: str):
      """从指定路径重新加载策略。"""
      _ckpt_runner.load(
        path,
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
      )
      return _ckpt_runner.get_inference_policy(device=device)

    if cfg.wandb_run_path is None:
      # 本地检查点模式:从本地目录获取可用检查点
      ckpt_dir = resume_path.parent

      def fetch_available_local() -> list[tuple[str, str]]:
        """获取本地可用的检查点列表。"""
        now = _time.time()
        entries: list[tuple[str, str, int]] = []
        for f in sorted(ckpt_dir.glob("*.pt")):
          try:
            step = int(f.stem.split("_")[1])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(int(now - f.stat().st_mtime))
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_local,
        load_checkpoint=lambda name: _reload_policy(str(ckpt_dir / name)),
      )
    else:
      # WandB检查点模式:从WandB获取可用检查点
      import wandb

      api = wandb.Api()
      run_path = str(cfg.wandb_run_path)
      wandb_run = api.run(run_path)
      _log_root = log_root_path  # pyright: ignore[reportPossiblyUnboundVariable]

      def fetch_available_wandb() -> list[tuple[str, str]]:
        """获取WandB运行中的可用检查点列表。"""
        wandb_run.load()
        now = datetime.now(tz=timezone.utc)
        entries: list[tuple[str, str, int]] = []
        for f in wandb_run.files():
          if not f.name.endswith(".pt"):
            continue
          try:
            step = int(f.name.split("_")[1].split(".")[0])
          except (IndexError, ValueError):
            step = 0
          ago = format_time_ago(
            int((now - _parse_wandb_dt(f.updated_at)).total_seconds())
          )
          entries.append((f.name, ago, step))
        entries.sort(key=lambda x: x[2])
        return [(name, t) for name, t, _ in entries]

      ckpt_manager = CheckpointManager(
        current_name=resume_path.name,
        fetch_available=fetch_available_wandb,
        load_checkpoint=lambda name: _reload_policy(
          str(get_wandb_checkpoint_path(_log_root, Path(run_path), name)[0])
        ),
        run_name=_parse_wandb_dt(wandb_run.created_at).strftime("%Y-%m-%d_%H-%M-%S"),
        run_url=wandb_run.url,
        run_status=wandb_run.state,
      )

  # 处理"auto"查看器选择
  if cfg.viewer == "auto":
    # 检查是否有显示环境变量来判断是否支持图形界面
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  try:
    # 启动选定的查看器
    if resolved_viewer == "native":
      # 原生MuJoCo查看器
      NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
      # Viser网络查看器
      ViserPlayViewer(env, policy, checkpoint_manager=ckpt_manager).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
  finally:
    # 清理:保存动作轨迹并关闭环境
    if trace_policy is not None:
      trace_policy.save()
    env.close()


def main():
  """主入口点:解析命令行参数并运行播放脚本。"""
  maybe_print_top_level_help("play")

  # 解析第一个参数以选择任务
  # 导入任务以填充注册表
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # 解析其余参数
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
