# DML-IL

This repository includes the implementation of the method DML-IL proposed in the paper "A Unifying Framework for Causal Imitation Learning with Hidden Confounders".

## Dependencies
The main dependencies of this codebased include:

- Pytorch: '2.4.1'

- stable_baselines3: '2.3.2'

- Mujoco: 210

- gymnasium: '0.29.1'

Our Cuda version is 11.8 with Python 3.8

## Basic Usage

Run experiments using DML-IL on the ticket pricing environment and Mujoco environments

```
python toy_main.py
python mujoco_main.py
```

The learner.py scripts includes the code for training; train_mujoco.py includes code for training the expert policies, with pre-trained experts in "experts" folder; models includes the neural network architecture and models.
