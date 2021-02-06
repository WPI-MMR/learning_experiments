from stable_baselines.common import callbacks

import wandb


class WandbRolloutCallback(callbacks.BaseCallback):
  def __init__(self, verbose=0):
    super().__init__(verbose=verbose)
    self.rollout_count = 0
  
  def _on_rollout_end(self) -> None:
    self.rollout_count += 1
    wandb.log({'rollouts': self.rollout_count}, step=self.num_timesteps)