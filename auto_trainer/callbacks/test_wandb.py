import unittest

from unittest import mock


class TestWandbEvalAndRecord(unittest.TestCase):
  def setUp(self):
    self.mock_env = mock.MagicMock()
    self.eval_episodes = 10
    self.render_freq = 2
    self.fps = 30
    self.wandb = mock.MagicMock()

    with mock.patch.dict('sys.modules', {'wandb': self.wandb}):
      from auto_trainer.callbacks import wandb as wandb_cb
      self.cb = wandb_cb.WandbEvalAndRecord(
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


if __name__ == '__main__':
  pass