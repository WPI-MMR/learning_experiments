import unittest

from auto_trainer import trainer
from unittest import mock

from wandb.integration.sagemaker import config


class TestAutoTrainer(unittest.TestCase):
  def test_synced_config_no_wandb(self):
    param = {'test_key': 'test_value'}

    trainer._WANDB = False
    self.assertFalse(trainer._WANDB)

    config, run = trainer.get_synced_config(param, ['bunch', 'of', 'tags'])
    
    self.assertDictEqual(config, param)
    self.assertIsNone(run)

  @mock.patch('wandb.init')
  def test_synced_config_wandb(self, mock_wandb):
    param = {'test_key': 'test_value'}
    tags = ['bunch', 'of', 'tags']

    mock_run = mock.Mock(config=param)
    mock_wandb.return_value = mock_run

    trainer._WANDB = True
    self.assertTrue(trainer._WANDB)
    
    config, run = trainer.get_synced_config(param, tags)

    self.assertDictEqual(config, param)
    self.assertEqual(run, mock_run)


if __name__ == '__main__':
  pass