from datetime import datetime
import stable_baselines
import logging
import os


PROJECT_NAME = 'solo-rl-experiments'
ENTITY = 'wpi-mmr'

SUPPORTED_ALGORITHMS = {
  'PPO2': stable_baselines.PPO2,
}


_DEFAULT_RUN_NAME = 'run'


try:
  import wandb
  _WANDB = True
except ImportError:
  logging.info('W&B Logging Disabled')
  _WANDB = False



def train(env, parameters, tags, full_logging=True, log_freq=100):
  config = parameters
  run = None

  if _WANDB:
    run = wandb.init(
      project=PROJECT_NAME,
      entity=ENTITY,
      config=parameters,
      tags=tags,
      sync_tensorboard=True,
    )
    config = run.config

  model_cls = SUPPORTED_ALGORITHMS[config.algorithm]
  model = model_cls(config.policy, env, 
                    tensorboard_log=_WANDB and wandb.run.dir,
                    full_tensorboard_log=full_logging,
                    verbose=1)

  if _WANDB: wandb.tensorboard.monkeypatch._notify_tensorboard_logdir(
      os.path.join(run.dir, '{}_1'.format(_DEFAULT_RUN_NAME)))

  model.learn(config.episodes, tb_log_name=_DEFAULT_RUN_NAME,
              log_interval=log_freq)
  model.save(_WANDB and os.path.join(wandb.run.dir, 'model') or datetime.now().strftime(
    '%m%d%H%M%S'))

  if _WANDB: run.finish()

  return model, config, run