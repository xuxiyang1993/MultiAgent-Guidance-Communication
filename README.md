# MultiAgent-Guidance-Communication

A Python implementation of the algorithm proposed in paper "Multi-Agent Autonomous Operations in Urban Air Mobility with Communication Constraints"

## Requirements

* python 3.6
* numpy
* gym
* ...


## Getting Started

Make sure that you have the above requirements taken care of, then download all the files. You can run it using

```
cd MCTS
python Agent_vertiport.py
```

Optional arguments:

`--save_path` the path where you want to save the output

`--seed` seed of the experiment

`--render` if render the env while running exp

`--decentralized` set it to True for decentralized control, default False

## MCTS Algorithm
The whole MCTS algorithm files are under the directory `MCTS\`

`common.py` defines the general MCTS node class and state class

`nodes_multi.py` defines the MCTS node class and state class specifically for Multi Agent Aircraft Guidance proble, e.g., given current aircraft state and current action, how to decided the next aircraft state

`search_multi.py` describes the search process of MCTS algorithm

## Simulator
The simulator codes are under `Simulators\`

`config_vertiport.py` defines the configurable parameters of the simulator. For example, airspace width/length, number of aircraft, scale (1 pixel = how many meters), conflict/NMAC distance, cruise/max speed of aircraft, heading angle change rate of aircraft, number simulations and search depth of MCTS algorithm, vertiport location, ...

`MultiAircraftVertiportEnv.py` is the main simulator. Some main functions include:

* `__init__`


If you have any questions or comments, don't hesitate to send me an email! I am looking for ways to make this code even more computationally efficient.

Email: xuxiyang@iastate.edu
