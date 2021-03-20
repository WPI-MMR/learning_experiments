from setuptools import setup, find_packages    

setup(name='auto_trainer', 
      version='0.0.1',
      python_requires='~=3.7',
      packages=find_packages(),
      install_requires=[ 
        'stable-baselines', 
        'numpy<1.19.0,>=1.16.0',
        'jupytext',
        'gym'
      ], 
      extras_require={
        'cpu': ['tensorflow>=1.15.0,<2'],
        'gpu': ['tensorflow-gpu==1.15.4'],
        'wandb': ['wandb', 'moviepy', 'imageio', 'plotly'],
        'mpi': ['stable-baselines[mpi]'],
        'gym': ['pyvirtualdisplay', 'PyOpenGL', 'PyOpenGL-accelerate']
      }
)