# DualML #
Accompanying repository for: 

Goulet Coulombe, Philippe and Göbel, Maximilian and Klieber, Karin (2024): "Dual Interpretation of Machine Learning Forecasts"

<br>

<hr>

## How to use this Reporsitory

Having trained his very own proprietry model, the researcher can use the function ``DualML`` to extract **observation weights**, **observation contributions**, and the corresponding metrics (**forecast concentration, short position, leverage, turnover**), as described in the paper.

On top of that, we also provide some example code (``DualML_run_inflation.R``), which is a replication of our inflation application for horizon ``h=1``.

<hr>

## Brief Overview of Files in this Repository ##

<br>

**ReadME_DualML.txt**

ReadMe on how to use the function ``DualML``.

<br>

**DualML.R**

Contains the main function ``DualML``, which generates observation weights, observation contributions and accompanying evaluation metrics (forecast concentration, short position, leverage, turnover).

Currently implemented models:

 - 'OLS':&emsp;any type of model that was fitted via OLS: $\quad \hat{\beta} = (X^\mathrm{T} X)^{-1} X^\mathrm{T}y$

 - 'RF':&emsp;&ensp;&nbsp;an object of type ``ranger``

 - 'LGB':&emsp;an object of type ``lightgbm``

 - 'RR':&emsp;&ensp;&nbsp;a ridge-regression model: $\quad \hat{\beta} = (X^\mathrm{T}X + \lambda*I_N)^{-1} X^\mathrm{T}y$

 - 'KRR':&emsp;a kernel-based ridge-regression model, in which forecasts are generated as: $\quad \hat{y} = K\left(K + \lambda*I_N\right)^{-1} y$


**Note**: our 'NN' is not included in the example code as it does not featured in our example code, but will be shipped with a future version, once I have found the time to wrap all this into a proper package.

<br>
    
**DualML_run_inflation.R**

Example code of the application: inflation forecasting at horizon ``h=1``.

<br>
  
**US_data.csv**

Data file for running the example in ``DualML_run_inflation.R``.


