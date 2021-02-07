import unittest

from unittest import mock

import importlib
import sys


class TestWandbEvalAndRecord(unittest.TestCase):
  def setUp(self):
    # TODO: Create a parent test case that encompasses this W&B mocking logic
    if 'wandb' in sys.modules:
      import wandb
      del wandb

    self.mock_env = mock.MagicMock()
    self.eval_episodes = 10
    self.render_freq = 2
    self.fps = 30
    self.wandb = mock.MagicMock()

    with mock.patch.dict('sys.modules', {'wandb': self.wandb}):
      import auto_trainer.callbacks.wandb 
      importlib.reload(auto_trainer.callbacks.wandb)

      self.cb = auto_trainer.callbacks.wandb.WandbEvalAndRecord(
        self.mock_env, self.eval_episodes, self.render_freq, self.fps)

  @mock.patch('numpy.transpose')
  @mock.patch('auto_trainer.callbacks.wandb.evaluate_policy')
  def test_step(self, mock_eval, mock_transpose):
    mean_reward = 69
    std_reward = 420
    mock_eval.return_value = mean_reward, std_reward
    
    self.cb.model = mock.MagicMock()
    self.cb.model.predict.return_value = None, None

    # Create an episode with length 10
    step_return_vals = [(None, None, False, None)] * 9
    step_return_vals.append((None, None, True, None))
    self.mock_env.step.side_effect = step_return_vals

    self.assertTrue(self.cb._on_step())

    self.assertEqual(
      len(self.mock_env.step.call_args_list), 10)
    self.assertEqual(
      len(self.mock_env.render.call_args_list), 10 / self.render_freq)
    
    self.wandb.log.assert_called_once()
    log = self.wandb.log.call_args[0][0]
    self.assertEqual(log['test_reward_mean'], mean_reward)
    self.assertEqual(log['test_reward_std'], std_reward)

    print(self.wandb)


class TestWandbRolloutsCallback(unittest.TestCase):
  def setUp(self):
    if 'wandb' in sys.modules:
      import wandb
      del wandb

    self.wandb = mock.MagicMock()

    with mock.patch.dict('sys.modules', {'wandb': self.wandb}):
      import auto_trainer.callbacks.wandb
      importlib.reload(auto_trainer.callbacks.wandb)

      self.cb = auto_trainer.callbacks.wandb.WandbRecordRollouts()

  def test_step_vs_rollout(self):
    self.assertEqual(self.cb.rollouts, 0)
    for _ in range(100):
      self.cb._on_step()
      self.cb.num_timesteps += 1

    self.assertEqual(self.cb.rollouts, 0)
    self.assertEqual(self.cb.num_timesteps, 100)

    for _ in range(10):
      self.cb.on_rollout_start()
      self.cb.on_rollout_end()
    self.assertEqual(self.cb.rollouts, 10)

    calls = self.wandb.log.call_args_list
    self.assertEqual(len(calls), 10)

    for i, (args, kwargs) in enumerate(calls):
      self.assertDictEqual(args[0], {'rollouts': i + 1})
      self.assertDictEqual(kwargs, {'step': 100})
      

if __name__ == '__main__':
  pass