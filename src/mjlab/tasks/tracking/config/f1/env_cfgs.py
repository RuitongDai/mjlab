"""Unitree f1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  F1_ACTION_SCALE,
  get_chair_cfg,
  get_f1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.event_manager import EventTermCfg


def f1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create f1 flat terrain tracking configuration."""
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_f1_robot_cfg(),
                        "chair": get_chair_cfg(),
                        }

  # 关闭自碰撞惩罚，避免坐姿/起身初期产生巨大负奖励。
  cfg.rewards.pop("self_collisions", None)
  # 多环境时必须在 reset event 里按 env_origins 重置 mocap pose
  cfg.events["reset_scene_to_default"] = EventTermCfg(
    func=envs_mdp.reset_scene_to_default,
    mode="reset",
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = {
    **F1_ACTION_SCALE,
    ".*_hip_pitch_joint": F1_ACTION_SCALE[".*_hip_pitch_joint"] * 4.0,
    ".*_knee_joint": F1_ACTION_SCALE[".*_knee_joint"] * 4.0,
    ".*_ankle_pitch_joint": F1_ACTION_SCALE[".*_ankle_pitch_joint"] * 2.0,
  }

  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
  )

  # reset模式设置为第0帧出生
  motion_cmd.sampling_mode = "start"

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )
  # 放宽terminations条件
  cfg.terminations["anchor_pos"].params["threshold"] = 1.0
  cfg.terminations["anchor_ori"].params["threshold"] = 1.8
  cfg.terminations["ee_body_pos"].params["threshold"] = 1.0

  cfg.viewer.body_name = "torso_link"

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_actor_terms = {
      k: v
      for k, v in cfg.observations["actor"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["actor"] = ObservationGroupCfg(
      terms=new_actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
