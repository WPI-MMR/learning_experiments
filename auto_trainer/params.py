from auto_trainer import trainer
import argparse

from typing import List


PROJECT_NAME = "solo-rl-experiments"
ENTITY = "wpi-mmr"


class BaseParameters:
  def parse(self):
    parser = argparse.ArgumentParser()
    parser = self.add_args(parser)
    args, _ = parser.parse_known_args()
    return args

  def add_args(self, parser: argparse.ArgumentParser):
    parser.add_argument('--episodes', default=10000,
                        help='number of episodes to train on')
    parser.add_argument('--episode_length', default=100000,
                        help='number of steps per episode')
    parser.add_argument('--policy', default='MlpPolicy',
                        help='which policy to train with')
    parser.add_argument('--algorithm', default='PPO2',
                        help='which algorithm to train on',
                        choices=list(trainer.SUPPORTED_ALGORITHMS.keys()))
    parser.add_argument('--num_workers', default=1,
                        help='how many parallel environments to use')
    return parser


class WandbParameters(BaseParameters):
  def add_args(self, parser: argparse.ArgumentParser):
    parser = super().add_args(parser)

    parser.add_argument('--eval_episodes', default=3,
                        help='number of episodes to evaluate the policy on')
    parser.add_argument('--eval_frequency', default=5,
                        help='after how many episodes the policy should be '
                             'evaluated')
    parser.add_argument('--eval_render_freq', default=50, help='after how many'
                        'steps should a render be taken of the env')
    parser.add_argument('--fps', default=10, help='fps to render the sim '
                                                  'images')

    return parser