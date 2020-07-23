# Optical RL-Gym

In this document we maintain a list of examples. This list will be updated with new resources.

## Examples using agents from Stable Baselines

We have a few examples of how to train agents using the Stable Baselines available agents [here](./stable_baselines/).

**List of examples:**

1. [DeepRMSA](./stable_baselines/DeepRMSA.ipynb)
2. [QoS Constrained Routing Assignment](./stable_baselines/QoSConstrainedRA.ipynb)

**Utils**

1. [create_topology](create_topology.py): script to load a topology file into a NetworkX graph and save it to a binary file ready for use in the environments
2. [create_topology_rmsa](create_topology_rmsa.py): script similar to the previous one, but which also includes modulation format properties of the paths, appropriate for use with RMSA environments.
3. [graph_utils](graph_utils.py): set of functions to load topology files into NetworkX graphs and compute paths.
