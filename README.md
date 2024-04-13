# Leveraging Local and Global Matching in Graph Neural Networks for Seedless Graph Matching

This repository is the implementation of "Leveraging Local and Global Matching in Graph Neural Networks for Seedless Graph Matching".

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

The Shrec'16 dataset can be downloaded [here](https://vision.in.tum.de/~laehner/shrec2016/files/TOPKIDS_lowres.zip). 
Then, unzip the downloaded file, rename the folder 'low resolution' as 'low_resolution', and put this folder under the folder './data'.

## Training

To train the model(s) in the paper, run this command:

```
python trainER.py
```

Then, a file of the trained model, named 'SeedGNN-model-trained.pth', will be generated and stored in the folder './model'.

## Evaluation

* To evaluate our model on ER graphs, run:

```
python TestERs.py
```

* To evaluate the NTMA algorithm and Degree profile algorithm on ER graphs,run:
```
python TestERntma.py
```
and
```
python TestERdp.py
```
* To evaluate our model on Facebook networks, run:
```
python TestFB.py
```
* To evaluate our model on Shrec'16 dataset, run:
```
python TestShrec16.py
```
