# Leveraging Local and Global Matching in GNN-based Seedless Graph Matching

This repository is the implementation of "Leveraging Local and Global Matching in GNN-based Seedless Graph Matching".

## Requirements

* Python (>=3.8)
* PyTorch (>=1.2.0)
* PyTorch Geometric (>=1.5.0)
* Numpy (>=1.20.1)
* Scipy (>=1.6.2)
* seaborn (0.11.2)
* graspologic (>=1.0.0)


## Preparing Data

The data of facebook networks used in our paper can be downloaded [here](https://archive.org/download/oxford-2005-facebook-matrix/facebook100.zip). 
Then, unzip the downloaded file and put the folder 'facebook100' under the folder './data'.

## Training

To train the model(s) in the paper, run this command:

```
python TrainER.py
```
A file of the trained model, named as "LGM-GNN-new.pth", will be generated and stored in the folder './model'.

## Pre-trained Models

A file of a pre-trained model, named 'Seedless-prod-8.pth', is stored in the folder './model'.

## Evaluation

* To evaluate our model on ER graphs with varying correlation $s$ in Figure. 8 and Figure 9, run:
```
python TestERs.py
```

* To evaluate the NTMA algorithm and Degree profile algorithm on ER graphs with varying correlation $s$ in Figure 9, run:
```
python TestERntma.py
```
and
```
python TestERdp.py
```

* To evaluate our model on Facebook networks in Figrue 10, run:
```
python TestFB.py
```



