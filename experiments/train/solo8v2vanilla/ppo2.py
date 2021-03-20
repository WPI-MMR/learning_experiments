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
# # PPO2 on Solo8 v2 Vanilla w/ Fixed Timestamp
# Only use the time-based stopping criteria. This is more of a rudamentary test more than anything.

# %% [markdown]
# ## Ensure that Tensorflow is using the GPU

# %%
import tensorflow as tf
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")

# %% [markdown]
# ## Define Experiment Tags

# %%
TAGS = ['solov2vanilla', 'gpu', 'standing_task', 
        'unnormalized_actions']

# %% [markdown]
# # Import required libraries

# %%
from gym_solo.envs import solo8v2vanilla
from gym_solo.core import obs
from gym_solo.core import rewards
from gym_solo.core import termination as terms

import gym
import gym_solo

# %% [markdown]
# ## Parse CLI arguments and register w/ wandb

# %% [markdown]
# This experiment will be using the auto trainer to handle all of the hyperparmeter running

# %%
from auto_trainer import params
import auto_trainer

# %% [markdown]
# Give the robot a total of 10 seconds simulation time to learn how to stand.

# %%
episode_length = 2 / solo8v2vanilla.Solo8VanillaConfig.dt
episode_length

# %% [markdown]
# Create a basic config

# %%
config = params.WandbParameters().parse()

config.episodes = 12500
config.episode_length = episode_length

config.num_workers = 6
config.eval_frequency = 50
config.eval_episodes = 3
config.fps = 15

# Create a 3 second gif
config.eval_render_freq = int(config.episode_length / (3 * config.fps))

config

# %%
config, run = auto_trainer.get_synced_config(config, TAGS)
config


# %% [markdown]
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
#
# Note that the autotrainer requires that the training environment be a `VecEnv` and the testing environment be a standard `gym.Env` for multi-processing.
#
# For us personally, we find that the easiest way to handle this is to create a Stable Baselines `VecEnv` generator (example can be found [here](https://stable-baselines.readthedocs.io/en/master/guide/examples.html#multiprocessing-unleashing-the-power-of-vectorized-environments)) and use that to generate both the training and testing environments.
#
# We also like to link our generator with our W&B config so that we can dynamically change the environments based from the web interface. 
#
# A full example can be seen below:

# %%
def make_env(length):
    def _init():
        env_config = solo8v2vanilla.Solo8VanillaConfig()
        env = gym.make('solo8vanilla-v0', config=env_config, 
                       normalize_actions=False)

        env.obs_factory.register_observation(obs.TorsoIMU(env.robot))
        env.obs_factory.register_observation(obs.MotorEncoder(env.robot))

        env.reward_factory.register_reward(1, rewards.UprightReward(env.robot))
        env.termination_factory.register_termination(terms.TimeBasedTermination(length))
        return env
    return _init


# %% [markdown]
# ### Create the Envs
# Import the desired vectorized env

# %%
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import VecNormalize

# %% [markdown]
# Create training & testing environments

# %%
train_env = SubprocVecEnv([make_env(config.episode_length) 
                           for _ in range(config.num_workers)])
# train_env = VecNormalize(train_env, clip_reward = 1.)

test_env = make_env(config.episode_length)()

# %% [markdown]
# ## Learning
# And we're off!

# %%
model, config, run = auto_trainer.train(train_env, test_env, config, TAGS, 
                                        log_freq=1000, full_logging=False, run=run)

# %%
