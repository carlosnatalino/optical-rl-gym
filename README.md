# Optical RL-Gym

[OpenAI Gym](https://gym.openai.com/) is the de-facto interface for reinforcement learning environments.
Optical RL-Gym builds on top of OpenAI Gym's interfaces to create a set of environments that model optical network problems such as resource management and reconfiguration.
Optical RL-Gym can be used to quickly start experimenting with reinforcement learning in optical network problems.
Later, you can use the pre-defined environments to create more specific environments for your particular use case.

Please use the following bibtex:

```
@inproceedings{optical-rl-gym,
  title = {The {Optical RL-Gym}: an open-source toolkit for applying reinforcement learning in optical networks},
  author = {Carlos Natalino and Paolo Monti},
  booktitle = {International Conference on Transparent Optical Networks (ICTON)},
  year = {2020},
  location = {Bari, Italy},
  month = {July},
  pages = {Mo.C1.1},
  doi = {10.1109/ICTON51198.2020.9203239},
  url = {https://github.com/carlosnatalino/optical-rl-gym}
}
```

## Features

Across all the environments, the following features are available:

- Use of [NetworkX](https://networkx.github.io/) for the topology graph representation, resource and path computation.
- Uniform and non-uniform traffic generation.
- Flag to let agents proactively reject requests or not.
- Appropriate random number generation with seed management providing reproducibility of results.

## Content of this document

1. <a href="#installation">Installation</a>
2. <a href="#environments">Environments</a>
3. <a href="#examples">Examples</a>
4. <a href="#resources">Resources</a>
5. <a href="#contributors">Contributors</a>
6. <a href="#contact">Contact</a>

<a href="#installation"><h2>Installation</h2></a>

You can install the Optical RL-Gym with:

```bash
git clone https://github.com/carlosnatalino/optical-rl-gym.git
cd optical-rl-gym
pip install -e .
``` 

You will be able to run the [examples](#examples) right away.

You can see the dependencies in the [setup.py](setup.py) file.

**To traing reinforcement learning agents, you must create or install reinforcement learning agents. Here are some of the libraries containing RL agents:**
- [Stable-baselines3](https://stable-baselines3.readthedocs.io/)
- [TensorFlow Agents](https://www.tensorflow.org/agents)
- [ChainerRL](https://github.com/chainer/chainerrl)
- [OpenAI Baselines](https://github.com/openai/baselines) -- in maintenance mode

<a href="#environments"><h2>Environments</h2></a>

At this moment, the following environments are ready for use:

1. RWAEnv
2. RMSAEnv
3. DeepRMSA

More environments will be added in the near future.

<a href="#examples"><h2>Examples</h2></a>

Training a RL agent for one of the Optical RL-Gym environments can be done with a few lines of code.

For instance, you can use a [Stable Baselines](https://github.com/hill-a/stable-baselines) agent trained for the RMSA environment:

```python
# define the parameters of the RMSA environment
env_args = dict(topology=topology, seed=10, allow_rejection=False, 
                load=50, episode_length=50)
# create the environment
env = gym.make('RMSA-v0', **env_args)
# create the agent
agent = PPO2(MlpPolicy, env)
# run 10k learning timesteps
agent.learn(total_timesteps=10000)
```

We provide a set of [examples](./examples).

<a href="#resources"><h2>Resources</h2></a>

- Introductory paper `The Optical RL-Gym: an open-source toolkit for applying reinforcement learning in optical networks` (paper and video to be published soon).
- [List of publications using Optical RL-Gym](./docs/PUBLICATIONS.md)
- [How to implement your own algorithm](./docs/Implementation.md)

<a href="#contributors"><h2>Contributors</h2></a>

Here is a list of people who have contributed to this project:

- Igor M. de Araújo [[GitHub](https://github.com/igormaraujo/)]
- Paolo Monti [[Personal page](https://www.chalmers.se/en/staff/Pages/Paolo-Monti.aspx)]

<a href="#contact"><h2>Contact</h2></a>

This project is maintained by Carlos Natalino [[Twitter](https://twitter.com/NatalinoCarlos)], who can be contacted through carlos.natalino@chalmers.se.
