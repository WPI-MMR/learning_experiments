from setuptools import setup


setup(name='auto_trainer',
      version='0.0.1',
      install_requires=['gym_solo', 'tensorflow-gpu', 'stable-baselines[mpi]'] 
)