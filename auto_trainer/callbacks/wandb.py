from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common import callbacks

import matplotlib.pyplot as plt
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
    rewards = []
    actions = []
    obses = []
    step_cnt = 0
    done, state = False, None
    obs = self.env.reset()
    while not done:
      if step_cnt % self.render_freq == 0:
        images.append(self.env.render(mode='rgb_array'))

      action, state = self.model.predict(obs, state=state, deterministic=True)
      obs, reward, done, _ = self.env.step(action)

      rewards.append(reward)
      actions.append(action)
      obses.append(obs)
      step_cnt += 1

    render = np.array(images)
    render = np.transpose(render, (0, 3, 1, 2))

    actions = np.array(actions).flatten()
    observes = np.array(obses).flatten()

    rewards = np.array(rewards)
    plt.clf()
    plt.plot(np.arange(len(rewards)), rewards)
    plt.xlabel('timesteps')
    plt.ylabel('rewards')
    plt.title('Timestep {}'.format(self.num_timesteps))

    wandb.log({
      'test_reward_mean': mean_rewards, 
      'test_reward_std': std_rewards,
      'render': wandb.Video(render, format='gif', fps=self.fps),
      'global_step': self.num_timesteps,
      'evaluations': self.n_calls,
      'reward_distribution': wandb.Histogram(rewards),
      'action_distribution': wandb.Histogram(actions),
      'observation_distribution': wandb.Histogram(observes),
      'reward_vs_time': wandb.Image(plt),
    }, step=self.num_timesteps)

    return True