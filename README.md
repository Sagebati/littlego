Shusaku
==

Go IA.


## Dependencies

libgoban


## Experiments and Results

### Supervised Learning

**Move Prediction Accuracy**

|                        | KGS train | KGS test | GoKifu validation |
| ---------------------- |:---------:|:--------:|:-----------------:|
| AlphaGo Zero           | 62.0      | 60.4     | 54.3              |
| AlphaGo                | 59.1      | 55.9     | -                 |
| LittleGo first_version | 43.8      | 37.0     | -                 |

**Game outcome prediction error**

|                        | KGS train | KGS test | GoKifu validation |
| ---------------------- |:---------:|:--------:|:-----------------:|
| AlphaGo Zero           | 0.177     | 0.185    | 0.207             |
| AlphaGo                | 0.19      | 0.37     | -                 |
| LittleGo first_version | 0.27      | 0.26     | -                 |

### Reinforcement Learning


**Move Prediction Accuracy**

|                        | KGS train | KGS test | GoKifu validation |
| ---------------------- |:---------:|:--------:|:-----------------:|
| AlphaGo Zero           | -         | -        | 49.0              |

**Game outcome prediction error**

|                        | KGS train | KGS test | GoKifu validation |
| ---------------------- |:---------:|:--------:|:-----------------:|
| AlphaGo Zero           | -         | -        | 0.177             |
