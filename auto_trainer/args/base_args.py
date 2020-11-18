from auto_trainer import utils
import argparse

from typing import List


PROJECT_NAME = "solo-rl-experiments"
ENTITY = "wpi-mmr"


class BaseModelConfiguration:
  def parse(self):
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
    parser.add_argument('--algorithm', default='PPO2',
                        help='which algorithm to train on',
                        choices=['PPO2', 'DRPO', 'TRPO'])
    return parser

  def add_args(self, parser: argparse.ArgumentParser):
    return parser