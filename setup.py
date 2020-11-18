from pkg_resources import DistributionNotFound, get_distribution
from setuptools import setup, find_packages    


setup(name='auto_trainer', 
      version='0.0.1',
      python_requires='~=3.7',
      install_requires=[ 
        'stable-baselines', 
        'jupytext'
      ], 
      extras_require={
        'cpu': ['tensorflow>=1.15.0,<2'],
        'gpu': ['tensorflow-gpu>=1.15.0,<2'],
        'wandb': ['wandb'],
      }
)