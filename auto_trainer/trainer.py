from typing import Any, Callable, List, Text

from datetime import datetime
from stable_baselines.common import callbacks as sb_cb
from stable_baselines.common.vec_env import VecEnv

import gym
import logging
import os
import stable_baselines


PROJECT_NAME = 'solo-rl-experiments'
ENTITY = 'wpi-mmr'

SUPPORTED_ALGORITHMS = {
  'PPO2': stable_baselines.PPO2,
}


try:
  import wandb
  _WANDB = True
except ImportError:
  logging.info('W&B Logging Disabled')
  _WANDB = False


def get_synced_config(parameters, tags: List[Text]):
  """Sync the config with wandb (if necessary) and return the new config

  Args:
    parameters (Any W&B supported type): The current hyperparameters to use. If
      W&B is enabled and is actively making a sweep, these hyperparameters will 
      get updated to W&B's sweep. This can be any type supported by W&B, 
      including Dicts and argparse.Namespace objects.
    tags (List[Text]): Tags that describe the run. Note that this is basically
      useless if W&B is disabled.

  Returns:
    The (hyperparameter config, W&B run object). Obviously, if W&B is disabled,
    then the run object will be None and the hyperparameter config will be what
    was passed in. 
  """
  if not _WANDB:
    return parameters, None

  run = wandb.init(
    project=PROJECT_NAME,
    entity=ENTITY,
    config=parameters,
    tags=tags,
  )
  config = run.config

  return config, run
  

def train(env: VecEnv, eval_env: gym.Env, parameters, tags: List[Text],
          full_logging: bool = False, log_freq: int = 100, run = None):
  """Train a model.

  Args:
    env (VecEnv): Vectorized gym environment to train on.
    eval_env (gym.Env): Gym environment to evaluate upon.
    parameters (Any W&B supported type): The hyperparameters to train with.
      Refer to W&B for all of the support types.
    tags (List[Text]): List of tags that describe this run. Doesn't do anything
      if `run` is not None.
    full_logging (bool, optional): Whether or not to log *everything*. Can fill
      up space quick. Defaults to True.
    log_freq (int, optional): How many steps to write the logs. Defaults to 100.
    run ([wandb.Run], optional): A current W&B run. Use this if you want to
      reuse a current run, i.e. train a model, do things to it, and continue
      training it. If this is None, a new run will be created via
      `get_synced_config`. Defaults to None.

  Returns:
    The model, configuration used, and the wandb run object (if applicable)
  """
  if run:
    config = parameters
  else: 
    config, run = get_synced_config(parameters, tags)

  model_cls = SUPPORTED_ALGORITHMS[config.algorithm]
  model = model_cls(config.policy, env, 
                    tensorboard_log=run.dir if _WANDB else './logs',
                    full_tensorboard_log=full_logging,
                    verbose=1)

  callbacks = []
  if _WANDB: 
    from auto_trainer.callbacks import wandb as wb_cb
    default_run_name = config.algorithm
    wandb.tensorboard.patch(
      save=True, 
      root_logdir=os.path.join(run.dir, '{}_1'.format(default_run_name)))

    eval_cb_raw = wb_cb.WandbEvalAndRecord(
      eval_env, config.eval_episodes, config.eval_render_freq, config.fps)
    eval_cb = sb_cb.EveryNTimesteps(
      n_steps=config.eval_frequency * config.episode_length, 
      callback=eval_cb_raw)

    callbacks.append(eval_cb)

  else:
    default_run_name = datetime.now().strftime('{}-%m%d%H%M%S'.format(
      config.algorithm))

  model.learn(total_timesteps=int(config.episodes * config.episode_length), 
              tb_log_name=default_run_name, log_interval=log_freq,
              callback=callbacks)
  model.save(_WANDB and os.path.join(run.dir, 'model') or \
    datetime.now().strftime('%m%d%H%M%S'))

  if _WANDB: run.finish()

  return model, config, run