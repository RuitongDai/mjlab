"""F1 robot MJCF, scene entity, and motion metadata."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab import MJLAB_SRC_PATH

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

F1_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "f1" / "xmls" / "f1_1.xml"
CHAIR_XML: Path = MJLAB_SRC_PATH / "asset_zoo" / "robots" / "f1" / "xmls" / "chair.xml"
assert F1_XML.exists()
assert CHAIR_XML.exists()


def get_spec() -> mujoco.MjSpec:
  # Empty spec.assets: MuJoCo resolves mesh files from disk (mjlab #873).
  return mujoco.MjSpec.from_file(str(F1_XML))


def get_chair_spec() -> mujoco.MjSpec:
  """Load and return a MuJoCo MjSpec for the 40 cm chair."""
  return mujoco.MjSpec.from_file(str(CHAIR_XML))


F1_ACTUATOR_LEGS = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_knee_joint"),
  stiffness=100.0,
  damping=2.0,
  effort_limit=75.0,
  armature=0.01,
)

F1_ACTUATOR_LEGS_HIPS_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=75.0,
  armature=0.01,
)

F1_ACTUATOR_FEET = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=75.0,
  armature=0.01,
)

F1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint", "waist_roll_joint"),
  stiffness=100.0,
  damping=2.0,
  effort_limit=50.0,
  armature=0.01,
)

F1_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
  ),
  stiffness=30.0,
  damping=2.0,
  effort_limit=25.0,
  armature=0.008,
)

F1_ACTUATOR_FORE_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_joint", ".*_wrist_roll_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=25.0,
  armature=0.005,
)

F1_ACTUATOR_HAND = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
  ),
  stiffness=20.0,
  damping=2.0,
  effort_limit=5.0,
  armature=0.005,
)
##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.86),
  joint_pos={
    ".*_hip_pitch_joint": -0.1,
    ".*_knee_joint": 0.2,
    ".*_ankle_pitch_joint": -0.1,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_elbow_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "right_shoulder_roll_joint": -0.0,
  },
  joint_vel={".*": 0.0},
)

SITTING_ON_CHAIR_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(-0.001838, -0.021784, 0.486345),
  joint_pos={
    "left_hip_pitch_joint": -1.363410,
    "left_hip_roll_joint": 0.240466,
    "left_hip_yaw_joint": 0.033856,
    "left_knee_joint": 1.639894,
    "left_ankle_pitch_joint": -0.087383,
    "left_ankle_roll_joint": 0.007715,
    "right_hip_pitch_joint": -1.375033,
    "right_hip_roll_joint": -0.089778,
    "right_hip_yaw_joint": 0.127244,
    "right_knee_joint": 1.654265,
    "right_ankle_pitch_joint": -0.103807,
    "right_ankle_roll_joint": 0.107536,
    "waist_yaw_joint": -0.015865,
    "waist_roll_joint": -0.029041,
    "left_shoulder_pitch_joint": 0.352869,
    "left_shoulder_roll_joint": 0.362515,
    "left_shoulder_yaw_joint": -0.359064,
    "left_elbow_joint": 0.008653,
    "left_wrist_roll_joint": 1.013690,
    "left_wrist_yaw_joint": 0.274577,
    "left_wrist_pitch_joint": 0.000000,
    "right_shoulder_pitch_joint": 0.313630,
    "right_shoulder_roll_joint": -0.295030,
    "right_shoulder_yaw_joint": 0.356980,
    "right_elbow_joint": 0.047584,
    "right_wrist_roll_joint": -1.083191,
    "right_wrist_yaw_joint": -0.279641,
    "right_wrist_pitch_joint": 0.000000,
  },
  joint_vel={".*": 0.0},
)

CHAIR_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  rot=(0.7071068, 0.0, 0.0, -0.7071068),
  joint_pos={},
  joint_vel={},
)

##
# Collision config.
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-7]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Entity config.
##

F1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    F1_ACTUATOR_LEGS,
    F1_ACTUATOR_LEGS_HIPS_PITCH,
    F1_ACTUATOR_FEET,
    F1_ACTUATOR_WAIST,
    F1_ACTUATOR_SHOULDER,
    F1_ACTUATOR_FORE_ARM,
    F1_ACTUATOR_HAND,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_f1_robot_cfg() -> EntityCfg:
  """Get a fresh F1 robot configuration instance."""
  # from wbc_mjlab.robots.f1.actuators import F1_ARTICULATION

  return EntityCfg(
    init_state=SITTING_ON_CHAIR_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=F1_ARTICULATION,
  )


def get_chair_cfg() -> EntityCfg:
  """Return a fresh fixed-base 40 cm chair configuration."""
  return EntityCfg(
    init_state=CHAIR_KEYFRAME,
    spec_fn=get_chair_spec,
  )


F1_ACTION_SCALE: dict[str, float] = {}
for a in F1_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    F1_ACTION_SCALE[n] = 0.25 * e / s

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_f1_robot_cfg())
  viewer.launch(robot.spec.compile())
