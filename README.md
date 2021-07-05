# Remaining Useful Life Prediction with LSTM   
PyTorch implementation of remaining useful life (RUL) prediction with LSTM, with evaluations on
NASA C-MAPSS engine data sets. Partially inspired by Zheng, S., Ristovski, K., Farahat, A., & Gupta, C. (2017, June). Long short-term memory network for remaining useful life estimation.   
_Author: Jiaxiang Cheng, Nanyang Technological University, Singapore_

<img alt="Python" src="https://img.shields.io/badge/python-%2314354C.svg?style=for-the-badge&logo=python&logoColor=white"/> <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" />

## Environment
```
python==3.8.10
numpy~=1.20.2
pandas~=1.2.5
matplotlib~=3.3.4
pytorch==1.9.0
```

## Usage
You may simply give the following command for both training and evaluation:
```
python main.py
```
Then you will get the following running information:
```
...

Epoch: 21, loss: 3076.69349, rmse: 27.08139
Epoch: 22, loss: 2955.86564, rmse: 24.61716
Epoch: 23, loss: 2841.80114, rmse: 23.69018
Epoch: 24, loss: 2779.35199, rmse: 23.40924

...
```
As the model and data sets are not heavy, the evaluation will be conducted after each
training epoch to catch up with the performance closely.
The prediction results will be stored in the folder ```_trials```.

## Citation & License
[![DOI](https://zenodo.org/badge/363314671.svg)](https://zenodo.org/badge/latestdoi/363314671)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
