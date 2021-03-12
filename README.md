# Solo 8 Learning Experiments
![build](https://github.com/WPI-MMR/learning_experiments/workflows/build/badge.svg?branch=master)
[![Coverage Status](https://coveralls.io/repos/github/WPI-MMR/learning_experiments/badge.svg?branch=master)](https://coveralls.io/github/WPI-MMR/learning_experiments?branch=master)

Experiments to try and get the solo 8 up and standing.

## Docker Execution Directions
As a lot of our training is gpu-dependant, we packaged our experiments to be
runnable on `nvidia-docker`. Therefore, to get the stack up and running, you
just need to build the included Dockerfiles (can be found in `bin/`). Note that
they don't require any files from the build context.

If you would like to run the stack using different versions of `gym_solo` 
and/or `learning_experiments`, you will need to replace their respective
volume mounts from within the container. They are both located within 
`/sources`. 

There is also a [convenience script](https://gist.github.com/agupta231/4e495cc4f34cfc6da018a2b65b01b675)
available to run the container with custom packages and open a jupyter instance
from within it.
