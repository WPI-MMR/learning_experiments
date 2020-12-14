import unittest

from auto_trainer import trainer
from unittest import mock


class TestAutoTrainer(unittest.TestCase):
  def test_synced_config_no_wandb(self):
    param = {'test_key': 'test_value'}

    trainer._WANDB = False
    self.assertFalse(trainer._WANDB)

    config, run = trainer.get_synced_config(param, ['bunch', 'of', 'tags'])
    
    self.assertDictEqual(config, param)
    self.assertIsNone(run)


if __name__ == '__main__':
  pass