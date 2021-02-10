# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Autotrainer PPO2 on Gym Pendulum
# Test the autotrainer on Open Ai's Pendulum environment, which is continuous and considered to be an easy enviroment to solve.

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
TAGS = ['gym-pendulum', 'gpu',]

# %% [markdown]
# ## Parse CLI arguments and register w/ wandb

# %% [markdown]
# This experiment will be using the auto trainer to handle all of the hyperparmeter running

# %%
from auto_trainer import params
import auto_trainer

auto_trainer.trainer.PROJECT_NAME = 'autotrainer-gym-baselines'

# %%
config = params.WandbParameters().parse()

config.episodes = 10000
config.episode_length = 750

config.num_workers = 8
config.eval_frequency = 25
config.eval_episodes = 5
config.fps = 20

# Create a 4 second gif
config.eval_render_freq = int(config.episode_length / (4 * config.fps))

config

# %%
config, run = auto_trainer.get_synced_config(config, TAGS)
config

# %% [markdown]
# ## Create a virtual display for environment rendering

# %%
import pyvirtualdisplay
display = pyvirtualdisplay.Display(visible=False, size=(1400, 900))
display.start()

# %% [markdown]
# ## Create a normalized wrapper for the Pendulum Environment
# The vanilla Pendulum enviornment has its action and observation spaces outside of $[-1, 1]$. Create a simple wrapper to apply min/max scaling to the respective values. Note that the default Pendulum environment doesn't have a termination state, so artifically create a termination condition.

# %%
from gym.envs.classic_control import pendulum
from gym import spaces
import gym

class NormalizedPendulum(pendulum.PendulumEnv):
    def __init__(self, length: int = 1000):
        super().__init__()
        self.unscaled_obs_space = self.observation_space
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,))
        self.observation_space = spaces.Box(low=-1., high=1., shape=(3,))
        
        self._length = length
        self._cnt = 0
    
    def reset(self):
        self._cnt = 0
        return super().reset()
    
    def step(self, u):
        self._cnt += 1
        
        obs, reward, done, info = super().step(u * self.max_torque)
        if self._cnt % self._length == 0:
            return obs, reward, True, info
        else:
            return obs, reward, done, info
    
    def _get_obs(self):
        return super()._get_obs() / self.unscaled_obs_space.high


# %% [markdown]
# Create the environment generator

# %%
def make_env(length):
    def _init():
        return NormalizedPendulum(length)
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
test_env = make_env(config.episode_length)()

# %% [markdown]
# ## Learning
# And we're off!

# %%
model, config, run = auto_trainer.train(train_env, test_env, config, TAGS, 
                                        log_freq=250, full_logging=False, run=run)

# %%
