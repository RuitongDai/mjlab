from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import f1_getup_tracking_env_cfg
from .rl_cfg import f1_getup_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Tracking-Getup-F1",
  env_cfg=f1_getup_tracking_env_cfg(),
  play_env_cfg=f1_getup_tracking_env_cfg(play=True),
  rl_cfg=f1_getup_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Getup-F1-No-State-Estimation",
  env_cfg=f1_getup_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=f1_getup_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=f1_getup_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
