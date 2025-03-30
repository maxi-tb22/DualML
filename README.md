# DualML #
Accompanying repository for: 

Goulet Coulombe, Philippe and Göbel, Maximilian and Klieber, Karin (2024): [Dual Interpretation of Machine Learning Forecasts](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5029492).

<br>



## How to use this Repository

Having trained his very own proprietry model, the researcher can use the function ``DualML`` to extract **observation weights**, **observation contributions**, and the corresponding metrics (**forecast concentration, short position, leverage, turnover**), as described in the paper.

On top of that, we also provide some example code (``DualML_run_inflation.R``), which is a replication of our inflation application for horizon ``h=1``.


**Note**: the current machinery only complies with regression-type models. The implementation for binary classification-type models will be made available soon.


## Brief Overview of Files in this Repository ##

<br>

**ReadME_DualML.txt**

ReadMe on how to use the function ``DualML``.

<hr>

<br>

**DualML.R**

Contains the main function ``DualML``, which generates observation weights, observation contributions and accompanying evaluation metrics (forecast concentration, short position, leverage, turnover).

Currently implemented models:

 - 'OLS':&emsp;any type of model that was fitted via OLS: $\quad \hat{\beta} = (X^\mathrm{T} X)^{-1} X^\mathrm{T}y$

 - 'RF':&emsp;&ensp;&nbsp;an object of type ``ranger``

 - 'LGB':&emsp;an object of type ``lightgbm``

 - 'RR':&emsp;&ensp;&nbsp;a ridge-regression model: $\quad \hat{\beta} = (X^\mathrm{T}X + \lambda*I_N)^{-1} X^\mathrm{T}y$

 - 'KRR':&emsp;a kernel-based ridge-regression model, in which forecasts are generated as: $\quad \hat{y} = K\left(K + \lambda*I_N\right)^{-1} y$

 - 'NN':&emsp;&ensp;an object that was fitted using the provided ``MLP.R``-function

<hr>

<br>

**MLP.R**

The function calling the 'NN'.

<hr>
    
**DualML_run_inflation.R**

Example code of the application: inflation forecasting at horizon ``h=1``.

<hr>

<br>
  
**US_data.csv**

Data file for running the example in ``DualML_run_inflation.R``.


