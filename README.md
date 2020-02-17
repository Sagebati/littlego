Shusaku
==

Go IA.


## Dependencies

libgoban


## Introduction

The goal of this project is to build an AI Agent for the game of Go based on AlphaGo Zero. The main idea is to reduce as most as possible the computation power needed and the computation time for that kind of agent by for example reducing the neural network architecture.

## Methodology

### Main differences between AlphaGo Zero and LittleGo

The neural network architecture consists in a "tower" of residual convolutionnal blocks where each block has 2 convolutionnnal layers and a residual connection connecting the input of the block. Then the network is separated in two heads, one for the prediction of move and the other for the prediction of the game outcome (evaluation function).

|                                 | AlphaGoZero            | LittleGo  |
| ------------------------------- |:----------------------:|:---------:|
| Number of residual blocks       | 20 or 40               | 5         |
| Number of convolutional filters | 256                    | 64        |
| Input planes                    | 17                     | 5         |
| Optimizer                       | RMSProp                | Adam      |
| Number of parameters            | 24,019,048 (20 blocks) <br> 47,642,728 (40 blocks) | 7,517,931 |


## Experiments and Results

### Supervised Learning

In order to validate their architecture DeepMind trains the neural network with supervised learning on the KGS dataset (https://u-go.net/gamerecords/) for training and testing set and GoKifu dataset (http://gokifu.com/) for validation set. Due to our computation limitation we were, for the moment, only able to train on a subset of the KGS dataset. 

|                         | KGS dataset size |
| ----------------------- |:----------------:|
| AlphaGo Zero            | ~ 30,000,000     |
| AlphaGo                 | ~ 30,000,000     |
| LittleGo                | 75,000           |
| LittleGo first_version  | 25,000           |

Below, the results tables for move prediction accuracy and game outcome prediction error, the important value to analyze is "KGS test".

#### Move prediction accuracy

|                         | KGS train | KGS test | GoKifu validation |
| ----------------------- |:---------:|:--------:|:-----------------:|
| AlphaGo Zero (20 block) | 62.0      | 60.4     | 54.3              |
| AlphaGo (12 layer)      | 59.1      | 55.9     | -                 |
| LittleGo                | 62.9      | 38.6     | -                 |
| LittleGo first_version  | 43.8      | 37.0     | -                 |

#### Game outcome prediction error

|                         | KGS train | KGS test | GoKifu validation |
| ----------------------- |:---------:|:--------:|:-----------------:|
| AlphaGo Zero (20 block) | 0.177     | 0.185    | 0.207             |
| AlphaGo (12 layer)      | 0.19      | 0.37     | -                 |
| LittleGo                | 0.15      | 0.195    | -                 |
| LittleGo first_version  | 0.27      | 0.26     | -                 |


### Reinforcement Learning

#### Move prediction accuracy

|                         | KGS train | KGS test | GoKifu validation |
| ----------------------- |:---------:|:--------:|:-----------------:|
| AlphaGo Zero (20 block) | -         | -        | 49.0              |
| AlphaGo Zero (40 block) | -         | -        | 51.3              |

#### Game outcome prediction error

|                         | KGS train | KGS test | GoKifu validation |
| ----------------------- |:---------:|:--------:|:-----------------:|
| AlphaGo Zero (20 block) | -         | -        | 0.177             |
| AlphaGo Zero (40 block) | -         | -        | 0.180             |
