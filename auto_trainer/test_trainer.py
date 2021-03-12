import unittest

from auto_trainer import trainer
from unittest import mock

import importlib
import sys


class TestAutoTrainer(unittest.TestCase):
  def test_synced_config_no_wandb(self):
    param = {'test_key': 'test_value'}

    trainer._WANDB = False
    self.assertFalse(trainer._WANDB)

    config, run = trainer.get_synced_config(param, ['bunch', 'of', 'tags'])
    
    self.assertDictEqual(config, param)
    self.assertIsNone(run)

  def test_synced_config_wandb(self):
    mock_wandb = mock.MagicMock()
    
    param = {'test_key': 'test_value'}
    tags = ['bunch', 'of', 'tags']

    mock_run = mock.MagicMock(config=param)
    mock_wandb.init = mock.MagicMock()
    mock_wandb.init.return_value = mock_run
    
    if 'wandb' in sys.modules: 
      import wandb
      del wandb
    with mock.patch.dict('sys.modules', {'wandb': mock_wandb}):
      importlib.reload(trainer)

      trainer._WANDB = True
      self.assertTrue(trainer._WANDB)
      config, run = trainer.get_synced_config(param, tags)

    mock_wandb.init.assert_called_once()

    _, kwargs = mock_wandb.init.call_args
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
    epi_length = 10

    parameters = mock.MagicMock(algorithm=algo, policy=policy, 
                                episodes=episodes, episode_length=epi_length)
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
      model, config, run = trainer.train(fake_env, fake_env, parameters, None)

      self.assertEqual(model, mock_model)
      self.assertEqual(config, parameters)
      self.assertIsNone(run)

      mock_model_cls.assert_called_once()
      args, _ = mock_model_cls.call_args
      self.assertTupleEqual(args, (policy, fake_env))

      mock_learn.assert_called_once()
      _, mock_learn_kwargs = mock_learn.call_args
      self.assertEqual(mock_learn_kwargs['total_timesteps'], 
                       episodes * epi_length)

      mock_save.assert_called_once()
      mock_save_args, _ = mock_save.call_args

      # If not using wandb, should save in 2-digit reps of 
      # MonthDayHourMinSec; henceforth, the length of the run name should be 10 
      # digits long
      self.assertEqual(len(mock_save_args[0]), 10)

  def test_trainer_wandb(self):
    algo = 'test_ago'
    policy = 'test_policy'
    episodes = 69

    parameters = mock.MagicMock(algorithm=algo, policy=policy, 
                                episodes=episodes)
    mock_env = mock.MagicMock()
    mock_run = mock.MagicMock(dir='test_dir')

    mock_learn = mock.MagicMock()
    mock_save = mock.MagicMock()
    mock_model = mock.MagicMock()
    mock_model.learn = mock_learn
    mock_model.save = mock_save

    mock_model_cls = mock.MagicMock()
    mock_model_cls.return_value = mock_model

    mock_wandb = mock.MagicMock()
    if 'wandb' in sys.modules: 
      import wandb
      del wandb
    with mock.patch.dict('sys.modules', {'wandb': mock_wandb}):
      importlib.reload(trainer)

      trainer._WANDB = True
      self.assertTrue(trainer._WANDB)
    
      with mock.patch.dict(trainer.SUPPORTED_ALGORITHMS, 
                          {algo: mock_model_cls}, clear=True):
        model, config, run = trainer.train(mock_env, mock_env, parameters, None, 
                                          run=mock_run)

    _, kwargs = mock_model_cls.call_args
    self.assertEqual(kwargs['tensorboard_log'], mock_run.dir)

    args, _ = mock_save.call_args
    self.assertEqual(args[0], '{}/model'.format(mock_run.dir))


if __name__ == '__main__':
  pass