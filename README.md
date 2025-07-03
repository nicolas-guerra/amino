### Neural Inverse Operators for solving PDE Inverse Problems
### A Measure-theoretic Inverse Neural Operator (AMINO)
This repository is the official implementation of the AMINO architecture in the paper [**Learning Where to Learn: Training Distribution Selection for Provable OOD Performance**](https://arxiv.org/abs/2505.21626). AMINO is a variant of [**Neural Inverse Operators for solving PDE Inverse Problems**](https://openreview.net/pdf?id=S4fEjmWg4X).

#### AMINO Architecture
<br/><br/>

<img src="Images/architecture.png" width="800" >

<br/><br/>

#### Requirements
The code is based on python 3 (version 3.7) and the packages required can be installed with
```
python3 -m pip install -r requirements.txt
```

#### Application
In this code, AMINO is applied specifically to the Caldéron problem with trigonometric coefficients.

#### Models Training
To train AMINO:
```
python3 RunAmino.py Example 0 data/Poisson.h5
```
All outputs will be saved in a created folder *Example*, load data in main process (denoted by 0), and the training and testing data used come from data/Poisson.h5. The models' hyperparameter can be specified in the corresponding python scripts as well.
