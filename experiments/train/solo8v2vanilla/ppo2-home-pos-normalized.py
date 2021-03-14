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
# # PPO2 on Solo8 v2 Vanilla for Quadrupedal Standing w/ a Multiplicitive Reward & Full Normalization
# Try to get the solo to stand on 4 feet stabley. Normalized both the action and observation spaces to fall between $[-1, 1]$

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
TAGS = ['solov2vanilla', 'gpu', 'home_pos_mulitiplicitive', 
        'normalized_actions', 'normalized_observations']

# %% [markdown]
# # Import required libraries

# %%
from gym_solo.envs import solo8v2vanilla
from gym_solo.core import obs
from gym_solo.core import rewards
from gym_solo.core import termination as terms

import gym
import gym_solo

import numpy as np

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

config.episodes = 4000
config.episode_length = episode_length

# Found experimentally - TODO make these into defaults
config.max_motor_rotation = np.pi / 2
config.flat_reward_hard_margin = 0.1
config.flat_reward_soft_margin = np.pi
config.height_reward_target = 0.33698
config.height_reward_hard_margin = 0.025
config.height_reward_soft_margin = 0.15
config.small_control_reward_margin = 10
config.hor_vel_reward_hard_margin = 0.5
config.hor_vel_reward_soft_margin = 3

config.num_workers = 4
config.eval_frequency = 1
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
# - How flat the torso is :$f$
# - Minimize the amount of control in the joints: $c$
# - Minimize the amount of torso movement: $m$
# - Keeping the torso at a given height: $h$
#
# We'll compose the "standing" reward to be $\frac{f + h}{2}$ as $f,h \in [0, 1]$. Then, the final reward becomes:
#
# $$reward = \frac{f + h}{2} cm$$
#
# Note that since $c,m \in [0, 1]$, this enforces that $reward \in [0, 1]$
#
# **Termination Criteria**
# - Terminate after $n$ timesteps

# %%
def make_env(length, max_motor_rot, fhm, fsm, stand_height, thm, tsm, scm, hmhm, hmsm):
    def _init():
        env_config = solo8v2vanilla.Solo8VanillaConfig()
        env_config.max_motor_rotation = max_motor_rot
        env = gym.make('solo8vanilla-v0', config=env_config, 
                    normalize_actions=True, 
                    normalize_observations=True)

        env.obs_factory.register_observation(obs.TorsoIMU(env.robot))
        env.obs_factory.register_observation(
            obs.MotorEncoder(env.robot, max_rotation=max_motor_rot))
        env.termination_factory.register_termination(terms.TimeBasedTermination(length))
        
        stand_reward = rewards.AdditiveReward()
        stand_reward.client = env.client
        
        stand_reward.add_term(0.5, rewards.FlatTorsoReward(
            env.robot, hard_margin=fhm, soft_margin=fsm))
        stand_reward.add_term(0.5, rewards.TorsoHeightReward(
            env.robot, stand_height, hard_margin=thm, soft_margin=tsm))
        
        home_pos_reward = rewards.MultiplicitiveReward(1, stand_reward,
            rewards.SmallControlReward(env.robot, margin=scm),
            rewards.HorizontalMoveSpeedReward(env.robot, 0,
                                            hard_margin=hmhm,
                                            soft_margin=hmsm))
        
        env.reward_factory.register_reward(1, home_pos_reward)
        return env
    return _init


# %% [markdown]
# ### Create the Envs
# Import the desired vectorized env

# %%
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.vec_env import VecNormalize

# %% [markdown]
# Load config and create the environments

# %%
envs = [make_env(config.episode_length, 
                config.max_motor_rotation,
                config.flat_reward_hard_margin,
                config.flat_reward_soft_margin,
                config.height_reward_target,
                config.height_reward_hard_margin,
                config.height_reward_soft_margin,
                config.small_control_reward_margin,
                config.hor_vel_reward_hard_margin,
                config.hor_vel_reward_soft_margin) 
        for _ in range(config.num_workers + 1)]

# %% [markdown]
# Create training & testing environments

# %%
train_env = SubprocVecEnv(envs[:-1])
test_env = envs[-1]()

# %% [markdown]
# ## Learning
# And we're off!

# %%
model, config, run = auto_trainer.train(train_env, test_env, config, TAGS, 
                                        log_freq=1000, full_logging=False, run=run)

# %%
