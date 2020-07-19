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
  url = {https://github.com/carlosnatalino/optical-rl-gym}
}
```

## Content of this document

1. <a href="#installation">Installation</a>
2. <a href="#environments">Environments</a>
3. <a href="#examples">Examples</a>
4. <a href="#resources">Resources</a>

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
- [Stable baselines](https://github.com/hill-a/stable-baselines)
- [OpenAI Baselines](https://github.com/openai/baselines) -- in maintenance mode
- [ChainerRL](https://github.com/chainer/chainerrl)
- [TensorFlow Agents](https://www.tensorflow.org/agents)

<a href="#environments"><h2>Environments</h2></a>

1. WDMEnv
2. EONEnv

<a href="#examples"><h2>Examples</h2></a>

We provide a growing set of [examples](./examples) 

<a href="#resources"><h2>Resources</h2></a>

- Introductory paper `The {Optical RL-Gym}: an open-source toolkit for applying reinforcement learning in optical networks` [[paper/pdf]()] [[presentation/pdf]()] [[presentation/YouTube]()].
- [List of publications using Optical RL-Gym](./docs/PUBLICATIONS.md)
