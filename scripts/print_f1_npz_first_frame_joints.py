"""Print the first-frame F1 root height and joint positions from a motion NPZ.

The F1 motion NPZ stores joint positions as a dense array without joint names.
This script uses the same joint order as ``mjlab.scripts.csv_to_npz_f1`` and
prints a dictionary that can be copied into ``SITTING_ON_CHAIR_KEYFRAME``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tyro

F1_JOINT_NAMES: tuple[str, ...] = (
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
)


def main(
  motion_file: Path = Path("datasets/npz/standup.npz"),
  frame: int = 0,
  precision: int = 6,
) -> None:
  """Print joint positions for one frame from an F1 motion NPZ."""
  if not motion_file.is_file():
    raise FileNotFoundError(f"Motion file not found: {motion_file}")

  data = np.load(motion_file)
  if "joint_pos" not in data:
    raise KeyError(f"{motion_file} does not contain a 'joint_pos' array")
  if "body_pos_w" not in data:
    raise KeyError(f"{motion_file} does not contain a 'body_pos_w' array")

  joint_pos = np.asarray(data["joint_pos"])
  body_pos_w = np.asarray(data["body_pos_w"])
  if joint_pos.ndim != 2:
    raise ValueError(
      f"Expected joint_pos shape (frames, joints), got {joint_pos.shape}"
    )
  if body_pos_w.ndim != 3 or body_pos_w.shape[2] != 3:
    raise ValueError(
      f"Expected body_pos_w shape (frames, bodies, 3), got {body_pos_w.shape}"
    )
  if body_pos_w.shape[0] != joint_pos.shape[0]:
    raise ValueError(
      f"body_pos_w frames ({body_pos_w.shape[0]}) do not match "
      f"joint_pos frames ({joint_pos.shape[0]})"
    )
  if joint_pos.shape[1] != len(F1_JOINT_NAMES):
    raise ValueError(
      f"Expected {len(F1_JOINT_NAMES)} F1 joints, got {joint_pos.shape[1]}"
    )
  if not 0 <= frame < joint_pos.shape[0]:
    raise ValueError(f"Frame index {frame} out of range [0, {joint_pos.shape[0] - 1}]")

  values = joint_pos[frame]
  root_pos = body_pos_w[frame, 0]
  fmt = f"{{:.{precision}f}}"

  print(f"# motion_file: {motion_file}")
  print(f"# joint_pos shape: {joint_pos.shape}")
  print(f"# body_pos_w shape: {body_pos_w.shape}")
  if "fps" in data:
    fps = np.asarray(data["fps"]).reshape(-1)[0]
    print(f"# fps: {float(fps):.3f}")
  print(f"# frame: {frame}")
  print(
    "root_pos="
    f"({fmt.format(float(root_pos[0]))}, "
    f"{fmt.format(float(root_pos[1]))}, "
    f"{fmt.format(float(root_pos[2]))}),"
  )
  print(f"root_height_z={fmt.format(float(root_pos[2]))}")
  print("joint_pos={")
  for name, value in zip(F1_JOINT_NAMES, values, strict=True):
    print(f'  "{name}": {fmt.format(float(value))},')
  print("},")


if __name__ == "__main__":
  tyro.cli(main, config=(tyro.conf.FlagConversionOff,))
