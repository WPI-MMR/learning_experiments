# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: 'Python 3.7.5 64-bit (''venv-solo-rl'': venv)'
#     language: python
#     name: python37564bitvenvsolorlvenvcc9eff967a5849f68175c6659045ec08
# ---

# %%
import auto_trainer.utils.safe_wandb

# %%
from auto_trainer import args

# %%
config = args.BaseModelConfiguration().get_args_for_run('test2', 'test2')

# %%
config

# %%
import gym
import gym_solo

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

from gym_solo.envs import solo8v2vanilla
from gym_solo.core import obs
from gym_solo.core import rewards
from gym_solo import testing

# %%
env_config = solo8v2vanilla.Solo8VanillaConfig()
env = gym.make('solo8vanilla-v0', config=env_config)
env.obs_factory.register_observation(obs.TorsoIMU(env.robot))
env.reward_factory.register_reward(1, rewards.UprightReward(env.robot))

# %%
import wandb

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log=wandb.run.dir)
model.learn(config.episodes)

# %%
