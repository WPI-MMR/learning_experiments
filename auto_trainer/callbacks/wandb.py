from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common import callbacks

import numpy as np
import wandb
import gym


class WandbEvalAndRecord(callbacks.BaseCallback):
  """W&B Evaluation and Recording Callback.
  
  This callback will evaluate the policy n times, render one episode, and
  report the mean & std of the reward and the rendering to W&B.
  """
  def __init__(self, env: gym.Env, eval_episodes: int, render_freq: int, 
               fps: int, verbose=0):
    """Create a new W&B Evalulation and Recording Callback.

    Args:
      env (gym.Env): The evaluation environment
      eval_episodes (int): How many episodes to evaluate over. Note a bigger
        number will cause slower training
      render_freq (int): Frequency to render the images. Lower will cause
        slower trainer times.
      fps (int): What fps to render the W&B gif at.
      verbose (int, optional): [description]. Defaults to 0.
    """
    super().__init__(verbose=verbose)
    self.env = env
    self.eval_episodes = eval_episodes
    self.render_freq = render_freq
    self.fps = fps

  def _on_step(self) -> bool:
    """Evaluate the current policy for self.eval_episodes, then take a render
    and report all stats to W&B

    Returns:
      True, as per API requirements
    """
    mean_rewards, std_rewards = evaluate_policy(
      self.model, self.env, n_eval_episodes=self.eval_episodes)
    
    images = []
    step_cnt = 0
    done, state = False, None
    obs = self.env.reset()
    while not done:
      if step_cnt % self.render_freq == 0:
        images.append(self.env.render())

      action, state = self.model.predict(obs, state=state, deterministic=True)
      obs, _, done, _ = self.env.step(action)
      step_cnt += 1

    render = np.array(images)
    render = np.transpose(render, (0, 3, 1, 2))

    wandb.log({
      'test_reward_mean': mean_rewards, 
      'test_reward_std': std_rewards,
      'render': wandb.Video(render, format='gif', fps=self.fps)
    }, step=self.num_timesteps)

    return True


class WandbRecordRollouts(callbacks.BaseCallback):
  """A simple callback to record the number of rollouts.
  
  This callback is useful because the majority of the recorded information
  is patched in from Tensorboard. This is fine, but with multi-processed
  environments, this causes a discrepency with the renderings and the 
  Tensoboard values.
  
  This creates a new standard x-axis called "rollouts" so that everything
  is called properly. Note that according to PPO2, rollouts should be the same
  as an episode or a trajectory.
  """
  def __init__(self, verbose: int = 0):
    super().__init__(verbose=verbose)
    self.rollouts = 0
    
  def _on_rollout_end(self) -> None:
    """Log the number of rollouts with the current timesteps
    """
    self.rollouts += 1
    wandb.log({'rollouts': self.rollouts}, step=self.num_timesteps)