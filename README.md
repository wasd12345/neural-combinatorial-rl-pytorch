# neural-combinatorial-rl-pytorch

PyTorch implementation of [Neural Combinatorial Optimization with Reinforcement Learning](https://arxiv.org/abs/1611.09940). 

This code is based on the pytorch-0.4 branch of Patrick E.'s neural-combinatorial-rl-pytorch: https://github.com/pemami4911/neural-combinatorial-rl-pytorch/tree/pytorch-0.4, with several modifications (RL algorithms, training, analysis, tasks).



## Running the code

Choose a task (e.g. "TSP_50" or "sort_10") in trainer.py.

For the TSP task, you'll need to download some validation data from the origin paper by running `python tsp_task.py`

Run the main code: `python trainer.py`

Output will be saved (data on the rewards during training). For the TSP task, figures of the learned tours will also be saved out.
 



## Adding other tasks

This implementation can be extended to support other combinatorial optimization problems. See the scripts in the "tasks" directory, e.g. `sorting_task.py`, `highlowhigh_task.py`, `tsp_task.py` for examples on how to add. The key thing is to provide a dataset class and a reward function that takes in a sample solution, selected by the pointer network from the input, and returns a scalar reward.

## Dependencies

* Python=3.6
* PyTorch=0.4


## Current Tasks

- TSP (2D symmetric Euclidean)

- sorting related tasks (sort, sort with random offset, sort nonconsecutive, high-low-high sort)



## Notes

- basic RL pretraining model with greedy decoding from the paper

- exponential moving average critic (critic network commented out)

- stochastic decoding policy in the pointer network during training, vs. beam search (**not yet finished**, only supports 1 beam a.k.a. greedy) for decoding when testing the model


## TODO

* [ ] Critic network vs. exponential moving average
* [ ] Range of sizes during training for problem instances (learned heuristics should transfer)
* [ ] Other combinatorial optimization problems [knapsack/bin packing, set cover]
* [ ] Finish implementing beam search decoding to support > 1 beam
* [ ] 2D -> N-Dim TSP [1) as fixed parameter e.g. D=3; 2) D=random variable over small single digit range [2,5] since some heuristics should transfer across dimensions]
* [ ] Multi-task? 
* [ ] Add RL pretraining-Sampling
* [ ] Add RL pretraining-Active Search
* [ ] Active Search