# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: 'Python 3.7.5 64-bit (''venv-solo-rl'': venv)'
#     language: python
#     name: python37564bitvenvsolorlvenvcc9eff967a5849f68175c6659045ec08
# ---

# %%
import auto_trainer.utils.safe_wandb

# %%
from auto_trainer import args

# %%
config = args.BaseModelConfiguration().get_args_for_run('test', 'test2')

# %%
config

# %%
