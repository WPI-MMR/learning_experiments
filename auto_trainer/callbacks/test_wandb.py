import unittest

from unittest import mock


class TestWandbRolloutCallback(unittest.TestCase):
  def setUp(self):
    with mock.patch.dict('sys.modules', unknown=mock.MagicMock()):
      from auto_trainer.callbacks import wandb as wandb_cb
      self.cb = wandb_cb.WandbRolloutCallback()

  def test_init(self):
    self.assertEqual(self.cb.rollout_count, 0)

  @mock.patch('wandb.log')
  def test_rollout(self, mock_log):
    for i in range(100):
      self.cb._on_step()
      self.cb.num_timesteps += 1

    self.assertEqual(self.cb.num_timesteps, 100)
    self.assertEqual(self.cb.rollout_count, 0)

    self.cb._on_rollout_end()
    self.assertEqual(self.cb.num_timesteps, 100)
    self.assertEqual(self.cb.rollout_count, 1)
    mock_log.assert_called_once_with({'rollouts': 1}, step=100)


if __name__ == '__main__':
  pass