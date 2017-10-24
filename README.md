# Baseline for variance reduction in Policy Gradient Algorithm  
Modular implementation of Vanila Policy Gradient (VPG) algorithm with baseline using an RNN policy.

# Dependencies
* Python 2.7 or 3.5
* [TensorFlow](https://www.tensorflow.org/) 1.10
* [gym](https://pypi.python.org/pypi/gym) 
* [numpy](https://pypi.python.org/pypi/numpy)
* [tqdm](https://pypi.python.org/pypi/tqdm) progress-bar

# Features
- Using a value function based baseline for reducing the variance in the vanila policy gradient algorithms
- Using an RNN policy for giving the action probabilities for a reinforcement learning problem
- Using a sampler that reshape the trajectory to be feed into an RNN policy
- Using gradient clipping to solve the exploding gradient problem
- Using GRU to solve the vanishing gradient problem  

# Usage

To train a model for Cartpole-v0:

	$ python test_graph_pg.py 

To view the tensorboard

	$tensorboard --logdir .
