from typing import List, Text

from datetime import datetime

import gym
import logging
import os
import stable_baselines


PROJECT_NAME = 'solo-rl-experiments'
ENTITY = 'wpi-mmr'
_DEFAULT_RUN_NAME = 'run'

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
      W&B is enabled and is actively making making a sweep, these 
      hyperparameters will get updated to W&B's sweep. This can be any type
      supported by W&B, including Dicts and argparse.Namespace objects.
    tags (List[Text]): Tags that describe the run. Note that this is basically
      useless if W&B is disabled.

  Returns:
    The (hyperparameter config, W&B run object). Obviously, if W&B is disabled,
    then the run object will be None and the hyperparameter config will be what
    was passed in. 
  """
  if not _WANDB:
    return parameters, None

  print('here')
  print(wandb)
  run = wandb.init(
    project=PROJECT_NAME,
    entity=ENTITY,
    config=parameters,
    tags=tags,
    sync_tensorboard=True,
  )
  config = run.config

  return config, run
  

def train(env: gym.Env, parameters, tags: List[Text], 
          full_logging: bool = False, log_freq: int = 100, run = None):
  """Train a model.

  Args:
    env (gym.Env): Gym environment to train on.
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
    [type]: [description]
  """
  if run:
    config = parameters
  else: 
    config, run = get_synced_config(parameters, tags)

  model_cls = SUPPORTED_ALGORITHMS[config.algorithm]
  model = model_cls(config.policy, env, 
                    tensorboard_log=_WANDB and run.dir,
                    full_tensorboard_log=full_logging,
                    verbose=1)

  if _WANDB: wandb.tensorboard.monkeypatch._notify_tensorboard_logdir(
    os.path.join(run.dir, '{}_1'.format(_DEFAULT_RUN_NAME)))

  model.learn(config.episodes, tb_log_name=_DEFAULT_RUN_NAME,
              log_interval=log_freq)
  model.save(_WANDB and os.path.join(run.dir, 'model') or datetime.now().strftime(
    '%m%d%H%M%S'))

  if _WANDB: run.finish()

  return model, config, run