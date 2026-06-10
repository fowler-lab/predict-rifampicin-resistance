# predict-rifampicin-resistance

The aim of this GitHub repository is to allow you to reproduce the results and figures of the following published manuscript

> Predicting rifampicin resistance in M. tuberculosis using machine learning informed by protein structural and chemical features
>
> Charlotte I. Lynch, Dylan Adlard, Philip W Fowler
> 
> ERJ Open Research (2024) 00952-2024 doi:[10.1183/23120541.00952-2024](https://doi.org/10.1183/23120541.00952-2024)

The repository contains three main juypter notebooks: `methods.ipynb`, `Results.ipynb`, and `Supplement.ipynb`.

To install the Python dependencies, you can either use `pip` 

```
$ pip install -r requirements.txt
```
or `conda` via

```
$ conda env create -f environment.yml 
$ conda activate predict_rif_ml
```
The latter is preferable since it will install all the dependencies in an environment.

