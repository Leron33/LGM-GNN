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




* To evaluate our model on ER graphs with varying correlation $s$ in Figure. 5 and Figure 6, run:
```
python TestERs.py
```

* To evaluate the FGNN, DGMC and NGM algorithm on ER graphs with varying correlation $s$ in Figure 1 and Figure 2, run:
```
python TestERfgnn.py
```
,
```
python TestERdgmc.py
```
and
```
python TestERngm.py

```

* To evaluate the NTMA, Degree profile, GRAMPA and FAQ algorithm on ER graphs with varying correlation $s$ in Figure 6, run:
```
python TestERntma.py
```
,
```
python TestERdp.py
```
,
```
python TestERgrampa.py
```
and
```
python TestERfaq.py
```



* To evaluate our model on Facebook networks in Figrue 7, run:
```
python TestFB.py
```

* T0 get Figure 9 and Figure 10, run:
```
python Layer_NTMA.py
```
and
```
python Layer_LGMGNN.py
```


