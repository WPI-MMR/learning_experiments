import unittest

from auto_trainer import trainer
from unittest import mock

from wandb.integration.sagemaker import config
from wandb.trigger import call


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

    mock_run = mock.MagicMock(config=param)
    mock_wandb.return_value = mock_run

    trainer._WANDB = True
    self.assertTrue(trainer._WANDB)
    
    config, run = trainer.get_synced_config(param, tags)
    mock_wandb.assert_called_once()

    _, kwargs = mock_wandb.call_args
    self.assertDictEqual(kwargs['config'], param)
    self.assertListEqual(kwargs['tags'], tags)
    self.assertDictEqual(config, param)
    self.assertEqual(run, mock_run)

  def test_trainer_no_wandb(self):
    trainer._WANDB = False
    self.assertFalse(trainer._WANDB)

    algo = 'test_ago'
    policy = 'test_policy'
    episodes = 69

    parameters = mock.MagicMock(algorithm=algo, policy=policy, 
                                episodes=episodes)
    fake_env = mock.MagicMock()

    mock_learn = mock.MagicMock()
    mock_save = mock.MagicMock()
    mock_model = mock.MagicMock()
    mock_model.learn = mock_learn
    mock_model.save = mock_save

    mock_model_cls = mock.MagicMock()
    mock_model_cls.return_value = mock_model

    with mock.patch.dict(trainer.SUPPORTED_ALGORITHMS, 
                         {algo: mock_model_cls}, clear=True):
      model, config, run = trainer.train(fake_env, parameters, None)

      self.assertEqual(model, mock_model)
      self.assertEqual(config, parameters)
      self.assertIsNone(run)

      mock_model_cls.assert_called_once()
      args, _ = mock_model_cls.call_args
      self.assertTupleEqual(args, (policy, fake_env))

      mock_learn.assert_called_once()
      mock_learn_args, _ = mock_learn.call_args
      self.assertTupleEqual(mock_learn_args, (episodes, ))

      mock_save.assert_called_once()
      mock_save_args, _ = mock_save.call_args

      # If not using wandb, should save in 2-digit reps of 
      # MonthDayHourMinSec, so the length of the run name should be 10 digits
      # long
      self.assertEqual(len(mock_save_args[0]), 10)


if __name__ == '__main__':
  pass