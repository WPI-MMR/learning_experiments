# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.9.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # PPO2 on Solo8 v2 Vanilla w/ Fixed Timestamp
# Only use the time-based stopping criteria. This is more of a rudamentary test more than anything.

# %% [markdown]
# ## Define Experiment Tags

# %%
TAGS = ['solov2vanilla', 'gpu']

# %% [markdown]
# ## Get Solo Environment Configuration

# %% [markdown]
# Import the relevant libraries + rewards & observations

# %%
from gym_solo.envs import solo8v2vanilla
from gym_solo.core import obs
from gym_solo.core import rewards
from gym_solo.core import termination as terms

import gym
import gym_solo

# %% [markdown]
# Create the config for the enviornment

# %%
env_config = solo8v2vanilla.Solo8VanillaConfig()

# %% [markdown]
# ## Parse CLI arguments and register w/ wandb

# %% [markdown]
# This experiment will be using the auto trainer to handle all of the hyperparmeter running

# %%
from auto_trainer import params
import auto_trainer

# %% [markdown]
# Create a basic config. Give the robot a total of 60 seconds simulation time to learn how to stand.

# %%
config = params.BaseParameters().parse()

config.episodes = 50000
config.episode_length = 60 / env_config.dt

config, run = auto_trainer.get_synced_config(config, TAGS)
config

# %% [markdown]
# ## Setup Environment
# Add the following inputs to the robot / environment:
#
# **Observations**
# - TorsoIMU
# - Motor encoder current values
#
# **Reward**
# - How upright the TorsoIMU is. Valued in $[-1, 1]$
#
# **Termination Criteria**
# - Terminate after $n$ timesteps

# %%
env = gym.make('solo8vanilla-v0', config=env_config)

env.obs_factory.register_observation(obs.TorsoIMU(env.robot))
env.obs_factory.register_observation(obs.MotorEncoder(env.robot))

env.reward_factory.register_reward(1, rewards.UprightReward(env.robot))

env.termination_factory.register_termination(terms.TimeBasedTermination(config.episode_length))

# %% [markdown]
# ## Learning

# %%
model, config, run = auto_trainer.train(env, config, TAGS, log_freq=500, 
                                        full_logging=False, run=run)

# %%
import pandas
from pympler import muppy, summary
all_objects = muppy.get_objects()
sum1 = summary.summarize(all_objects)
# Prints out a summary of the large objects
summary.print_(sum1)
