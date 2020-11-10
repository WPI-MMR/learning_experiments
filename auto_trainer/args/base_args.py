from auto_trainer import utils
import argparse

from typing import List


PROJECT_NAME = "solo-rl-experiments"
ENTITY = "wpi-mmr"


class BaseModelConfiguration:
  def get_args_for_run(self, run_name: str, run_tags: List[str]):
    wandb = utils.safe_import_wandb()

    if wandb:
      wandb.init(name=run_name,
                 project=PROJECT_NAME,
                 entity=ENTITY,
                 tags=run_tags,
                 config=self._parse())
      return wandb.config
    else:
      print('Not using wandb')
      return self._parse()

  def _parse(self):
    parser = argparse.ArgumentParser()
    parser = self._add_global_args(parser)
    parser = self.add_args(parser)

    args, _ = parser.parse_known_args()
    return args

  def _add_global_args(self, parser: argparse.ArgumentParser):
    parser.add_argument('--episodes', default=10000,
                        help='number of episodes to train on')
    parser.add_argument('--policy', default='MlpPolicy',
                        help='which policy to train with')
    return parser

  def add_args(self, parser: argparse.ArgumentParser):
    return parser