# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Auto Trainer & Stable Baselines Demostration w/ Solo8
# A quick example on how to use the auto trainer framework alongisde stable-baslines to quickly monitor and train a model. Note that we will be using Weights and Biases to make the monitoring a bit easier.

# %% [markdown]
# ## Define the experiment tags
# This isn't necessary if you aren't using W&B, but it's a great way to organize your run data!

# %%
TAGS = ['demo', 'solov2vanilla', 'cpu']

# %% [markdown]
# ## Set up the Solo Environment
# Note that this is using the Solov2Vanilla environment.

# %% [markdown]
# Import all of the required packages

# %%
from gym_solo.envs import solo8v2vanilla
from gym_solo.core import obs
from gym_solo.core import rewards
from gym_solo import testing

import gym
import gym_solo

# %% [markdown]
# Create the environment config and instaniate the registered environment

# %%
env_config = solo8v2vanilla.Solo8VanillaConfig()
env = gym.make('solo8vanilla-v0', config=env_config)

# %% [markdown]
# Register all of the observations and rewards. Note that in this case, the observation is just the IMU and the rewards is on how upright the robot is. Modifying these values will probably be the biggest factor in determining convergence.

# %%
env.obs_factory.register_observation(obs.TorsoIMU(env.robot))
env.reward_factory.register_reward(1, rewards.UprightReward(env.robot))

# %% [markdown]
# And now the environment is all prepped to train on!

# %% [markdown]
# ## Learning!
# As soon as the env is ready, learning *should* be trivial.

# %% [markdown]
# Import the parameters and the `auto_trainer` itself. The `BaseParameters` is an extendable class where you can ad custom parameters for specific experiments. In this case, we arne't doing anything special, so we will just use the barebones verison.

# %%
from auto_trainer.params import BaseParameters
import auto_trainer

# %% [markdown]
# Parse the base parameters. Note that if you are performing a Weights & Biases sweep, these values will get overriden by the global organizer.

# %%
config = BaseParameters().parse()
config.episodes=5000

config

# %% [markdown]
# And thats it, we should be ready to train!

# %%
model, config, run = auto_trainer.train(env, config, TAGS, log_freq=500)
