from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common import callbacks

import numpy as np
import wandb
import gym


class WandbEvalAndRecord(callbacks.BaseCallback):
  def __init__(self, env: VecEnv, eval_episodes, verbose=0, step_dt=0.01, 
               render_freq=50):
    super().__init__(verbose=verbose)
    self.env = env
    self.eval_episodes = eval_episodes
    self.render_freq = render_freq
    self.step_dt = step_dt

  def _on_step(self) -> bool:
    mean_rewards, std_rewards = evaluate_policy(
      self.model, self.env, n_eval_episodes=self.eval_episodes)
    
    images = []
    step_cnt = 0
    done, state = False, None
    obs = self.env.reset()
    while not done:
      if step_cnt % self.render_freq == 0:
        images.append(self.env.get_images()[0])

      action, state = self.model.predict(obs, state=state, deterministic=True)
      obs, _, done, _ = self.env.step(action)
      step_cnt += 1

    render = np.array(images)
    render = np.transpose(render, (0, 3, 1, 2))

    wandb.log({
      'Test Reward mean': mean_rewards, 
      'Test Reward std': std_rewards,
      'Render': wandb.Video(render, format='gif', fps=10)
    }, step=self.num_timesteps)