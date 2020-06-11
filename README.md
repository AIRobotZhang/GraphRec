# Graph Neural Networks for Social Recommendation
Data in social recommender systems can be represented as user-user social graph and user-item graph; and learning latent factors of users and items is the key. Graph Neural Networks can naturally integrate node information and topological structure which have been demonstrated to be powerful in learning on graph data. However, building social recommender systems based on GNNs faces challenges. For example, the user-item graph encodes both interactions and their associated opinions; social relations have heterogeneous strengths; users involve in two graphs (e.g., the user-user social graph and the user-item graph). To address the three aforementioned challenges simultaneously, the paper presented a novel graph neural network framework (GraphRec) for social recommendations. 

Paper is available at [this site](https://dl.acm.org/doi/pdf/10.1145/3308558.3313488).

#### For the purpose of fully understanding this framework and learning social recommender algorithms, we implemented this code.

## Requirements
* Python==3.7.4
* pytorch==1.3.1
* pickle
* numpy
* tqdm
* sklearn
* scipy

## Input data format
Ciao and Epinions Dataset can be available in dataset folder. The data format is as follows('\t' means TAB):

```
userid \t itemid \t categoryid \t rating \t helpfulness \t timestamp
...
```

## How to run
Train & Dev & Test:
Original dataset is split into training, validation and testing dataset according to the rating timestamp of each user. You can run the `preprocess.py` in data folder:
```
$ python preprocess.py
```
To train the model, run `run_GraphRec.py`:
```
$ python run_GraphRec.py --epoch 100 --batch_size 128 --embed_dim 300 --lr 0.001 --dataset ciao
```

More detailed configurations can be found in `config.py`, which is in utils folder.

## Reference
```
[1] Fan W, Ma Y, Li Q, et al. Graph neural networks for social recommendation. WWW 2019.
[2] https://github.com/wenqifan03/GraphRec-WWW19
```

## Disclaimer

The code is for research purpose only and released under the Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
