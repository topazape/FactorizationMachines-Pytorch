# FactorizationMachines-Pytorch
Factorization Machines for Criteo CTR Prediction Contest.

[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge) is a competition held on Kaggle in 2014, focused on predicting click-through rates (CTR) for online advertising. Provided by the French online advertising company Criteo, participants were tasked with developing a model to predict the click rates of advertisements using 10 days' worth of anonymized web click logs provided by Criteo.

## Dataset
The datasets used in this repository can be obtained from [criteo dataset](https://www.kaggle.com/datasets/mrkmakr/criteo-dataset).

## Model
```
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
FactorizationMachines                    390
├─Embedding: 1-1                         1,116,232
├─Linear: 1-2                            40
=================================================================
Total params: 1,116,662
Trainable params: 1,116,662
Non-trainable params: 0
=================================================================
```

## Usage
```shell
python run examples/config.toml --seed 42
```

## Result
### loss
![download](https://user-images.githubusercontent.com/38512143/227750637-d3484234-dd55-4003-bf17-ccfb70c72e37.png)
### ROCAUC
![download-1](https://user-images.githubusercontent.com/38512143/227750644-9fc45de7-1c4e-4ffd-842c-0a3049f917f1.png)
