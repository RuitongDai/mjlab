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
  target_names_expr=(".*_hip_yaw_joint", ".*_hip_roll_joint", ),
  stiffness=100.0,
  damping=2.0,
  effort_limit=87.0,
  armature=0.01,
)
F1_ACTUATOR_KNEE = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_knee_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=120.0,
  armature=0.01,
)

F1_ACTUATOR_LEGS_HIPS_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_pitch_joint",),
  stiffness=100.0,
  damping=2.0,
  effort_limit=75.0,
  armature=0.01,
)

F1_ACTUATOR_ANKLE_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=89.0,
  armature=0.01,
)
F1_ACTUATOR_ANKLE_ROLL = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_roll_joint",),
  stiffness=30.0,
  damping=2.0,
  effort_limit=12.0,
  armature=0.01,
)

F1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint", "waist_roll_joint"),
  stiffness=100.0,
  damping=2.0,
  effort_limit=87.0,
  armature=0.01,
)

F1_ACTUATOR_SHOULDER_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
  ),
  stiffness=30.0,
  damping=2.0,
  effort_limit=48.0,
  armature=0.008,
)
F1_ACTUATOR_SHOULDER = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
  ),
  stiffness=30.0,
  damping=2.0,
  effort_limit=84.0,
  armature=0.008,
)

F1_ACTUATOR_FORE_ARM = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_elbow_joint", ".*_wrist_roll_joint"),
  stiffness=30.0,
  damping=2.0,
  effort_limit=37.0,
  armature=0.005,
)

F1_ACTUATOR_WRIST_YAW = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_yaw_joint",
  ),
  stiffness=20.0,
  damping=2.0,
  effort_limit=20.0,
  armature=0.005,
)
F1_ACTUATOR_WRIST_PITCH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_wrist_pitch_joint",
  ),
  stiffness=20.0,
  damping=2.0,
  effort_limit=25.0,
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
  pos=(-0.047054, 0.016653, 0.513793),
  joint_pos={
    "left_hip_pitch_joint": -1.403895,
    "left_hip_roll_joint": 0.291679,
    "left_hip_yaw_joint": 0.204824,
    "left_knee_joint": 1.781702,
    "left_ankle_pitch_joint": -0.310334,
    "left_ankle_roll_joint": 0.100182,
    "right_hip_pitch_joint": -1.458333,
    "right_hip_roll_joint": -0.367897,
    "right_hip_yaw_joint": -0.140928,
    "right_knee_joint": 1.780921,
    "right_ankle_pitch_joint": -0.324074,
    "right_ankle_roll_joint": -0.150731,
    "waist_yaw_joint": 0.016074,
    "waist_roll_joint": -0.070227,
    "left_shoulder_pitch_joint": -0.199918,
    "left_shoulder_roll_joint": 0.391050,
    "left_shoulder_yaw_joint": 0.212286,
    "left_elbow_joint": 1.297245,
    "left_wrist_roll_joint": 0.829035,
    "left_wrist_yaw_joint": 0.104108,
    "left_wrist_pitch_joint": 0.001853,
    "right_shoulder_pitch_joint": -0.151481,
    "right_shoulder_roll_joint": -0.293825,
    "right_shoulder_yaw_joint": -0.074562,
    "right_elbow_joint": 1.087002,
    "right_wrist_roll_joint": -1.306453,
    "right_wrist_yaw_joint": -0.265385,
    "right_wrist_pitch_joint": 0.096061,
  },
  joint_vel={".*": 0.0},
)

GETUP_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.823084, -2.270562, 0.044188),
  rot=(0.709333, -0.104059, -0.680059, -0.153423),
  joint_pos={
    "left_hip_pitch_joint": -0.136787,
    "left_hip_roll_joint": 0.228250,
    "left_hip_yaw_joint": 0.565108,
    "left_knee_joint": 0.218349,
    "left_ankle_pitch_joint": 0.005376,
    "left_ankle_roll_joint": -0.226489,
    "right_hip_pitch_joint": -0.240797,
    "right_hip_roll_joint": -0.308974,
    "right_hip_yaw_joint": -0.203323,
    "right_knee_joint": 0.343597,
    "right_ankle_pitch_joint": -0.361462,
    "right_ankle_roll_joint": 0.082111,
    "waist_yaw_joint": 0.041346,
    "waist_roll_joint": -0.011172,
    "left_shoulder_pitch_joint": 0.335835,
    "left_shoulder_roll_joint": 1.066024,
    "left_shoulder_yaw_joint": 0.227511,
    "left_elbow_joint": 1.179013,
    "left_wrist_roll_joint": 0.247639,
    "left_wrist_yaw_joint": 0.661832,
    "left_wrist_pitch_joint": 0.219625,
    "right_shoulder_pitch_joint": 0.329745,
    "right_shoulder_roll_joint": -1.186348,
    "right_shoulder_yaw_joint": -0.379843,
    "right_elbow_joint": 1.241661,
    "right_wrist_roll_joint": 0.970049,
    "right_wrist_yaw_joint": -0.659664,
    "right_wrist_pitch_joint": -0.215926,
  },
)

CHAIR_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.0),
  rot=(1,0,0,0),
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

GETUP_SUPPORT_GEOMS = (
  r"^(left|right)_foot[1-7]_collision$",
  r"^(left|right)_hand_collision$",
  r"^(left|right)_wrist_collision$",
  r"^(left|right)_elbow_yaw_collision$",
  r"^(left|right)_shin_collision$",
  r"^(left|right)_thigh_collision$",
  r"^pelvis_collision$",
  r"^torso_collision$",
)
GETUP_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={
    **{name: 3 for name in GETUP_SUPPORT_GEOMS},
    ".*_collision": 1,
  },
  priority={name: 1 for name in GETUP_SUPPORT_GEOMS},
  friction={name: (0.8,) for name in GETUP_SUPPORT_GEOMS},
)

##
# Entity config.
##

F1_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    F1_ACTUATOR_LEGS,
    F1_ACTUATOR_KNEE,
    F1_ACTUATOR_LEGS_HIPS_PITCH,
    F1_ACTUATOR_ANKLE_PITCH,
    F1_ACTUATOR_ANKLE_ROLL,
    F1_ACTUATOR_WAIST,
    F1_ACTUATOR_WRIST_PITCH,
    F1_ACTUATOR_WRIST_YAW,
    F1_ACTUATOR_SHOULDER,
    F1_ACTUATOR_SHOULDER_PITCH,
    F1_ACTUATOR_FORE_ARM
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

def get_f1_getup_robot_cfg() -> EntityCfg:
  return EntityCfg(
    init_state= GETUP_KEYFRAME,
    collisions=(GETUP_FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=F1_ARTICULATION,
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
