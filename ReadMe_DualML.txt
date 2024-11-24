% ----------------------------------------------------------------------------- %
        	      ReadMe for function: DualML.R

				      based on

     Goulet Coulombe, Philippe, and Goebel, Maximilian, and Klieber, Karin (2024)

		
		"Dual Interpretation of Machine Learning Forecasts"% ----------------------------------------------------------------------------- %

Imagine you have fitted some kind of model -- e.g. an AR(p), a random forest, a boosting-algorithm, a ridge-regression, etc. -- and have generated the corresponding predictions. 

THIS FUNCTION allows you to insert the corresponding model constituents (feature matrix, hyperparameters, etc.) and receive observation weights, observation contributions, and corresponding metrics as described in the paper:



Currently implemented models:

 - 'OLS': 	any type of model that was fitted via OLS, for which coefficients can be estimated in the PRIMAL SPACE as: \hat{beta} = (X'X)^-1 X'y 

 - 'RF':	an object of type ``ranger``

 - 'LGB':	an object of type ``lightgbm``

 - 'RR':	a ridge-regression model, for which coefficients can be estimated in the PRIMAL SPACE as: \hat{beta} = (X'X + lmbda*I_N)^-1 X'y

 - 'KRR':	a kernel-based ridge-regression model, in which forecasts are generated as: K (K + lmbda*I_N)^-1 y

 - 'NN':	a object that was fitted using the provided ``MLP.R``-function



Currently implemented classes of models:

	- regression:		yes
	- classification:	still to come



% ======================= Required Inputs to the Function ====================== #


 - run__model:		a list (the equivalence of a python dictionary) with entries 'type' and 'params'


 	-- 'type':	the type of fitted model for which you want to receive observation weights/contributions, and corresponding metrics


 	-- 'params':	itself a list of model-specific parameters/arguments


	---> SEE BELOW FOR EXAMPLES OF ``run__model`` for the types of models that are currently implemented 


 - run__type:		the type of model class that was fitted

	--- Arguments:	'regression'; 'classification'


 - Q:			used for calculating Forecast Concentration and denotes the proportion of total weights attributed to the top Q% of the weights; default: Q=5  




% ======================= EXAMPLES OF ``run__model`` ====================== #


% ---------------------------- Type: OLS ---------------------------- %


run__model <- list(
			'type' = 'OLS',
			'params' = list(
					'Xtrain' = [an N x F matrix of features of your insample set],
					'Xtest' = [a T x F matrix of features of your out-of-sample set],
					'Ytrain' = [an N x 1 vector of observations of your target],
					'dates_ins' = [OPTIONAL: a vector of dates corresp[onding to your insample observations; if set to NULL the 'date' columns in the output matrices will be numbered as 1,...,nrow('Xtrain')],
					'model_object' = NULL,
					'lmbda' = NULL,
					'intercept' = [TRUE/FALSE: was the model fitted with an intercept or not?]
		  )



% ---------------------------- Type: RF ---------------------------- %


run__model <- list(
			'type' = 'RF',
			'params' = list(
					'Xtrain' = [an N x F matrix of features of your insample set],
					'Xtest' = [a T x F matrix of features of your out-of-sample set],
					'Ytrain' = [an N x 1 vector of observations of your target],
					'dates_ins' = [OPTIONAL: a vector of dates corresp[onding to your insample observations; if set to NULL the 'date' columns in the output matrices will be numbered as 1,...,nrow('Xtrain')],
					'model_object' = [an object of type ranger],
					'lmbda' = NULL
		  )



% ---------------------------- Type: LGB ---------------------------- %


run__model <- list(
			'type' = 'LGB',
			'params' = list(
					'Xtrain' = [an N x F matrix of features of your insample set],
					'Xtest' = [a T x F matrix of features of your out-of-sample set],
					'Ytrain' = [an N x 1 vector of observations of your target],
					'dates_ins' = [OPTIONAL: a vector of dates corresp[onding to your insample observations; if set to NULL the 'date' columns in the output matrices will be numbered as 1,...,nrow('Xtrain')],
					'model_object' = [an object of type lightgbm],
					'lmbda' = NULL
		  )


% ---------------------------- Type: RR ---------------------------- %


run__model <- list(
			'type' = 'RR',
			'params' = list(
					'Xtrain' = [an N x F matrix of features of your insample set],
					'Xtest' = [a T x F matrix of features of your out-of-sample set],
					'Ytrain' = [an N x 1 vector of observations of your target],
					'dates_ins' = [OPTIONAL: a vector of dates corresp[onding to your insample observations; if set to NULL the 'date' columns in the output matrices will be numbered as 1,...,nrow('Xtrain')],
					'model_object' = NULL,
					'lmbda' = [the L2-type penalty parameter],
					'intercept' = [TRUE/FALSE: was the model fitted with an intercept or not?]
		  )


% ---------------------------- Type: KRR ---------------------------- %


run__model <- list(
			'type' = 'KRR',
			'params' = list(
					'Xtrain' = [an N x N matrix of proximities (K = \Phi(Xtrain), where \Phi is some kernel function) of features in your insample set],
					'Xtest' = [a T x T matrix of proximities of features (K = \Phi(Xtest), where \Phi is some kernel function) in out-of-sample set],
					'Ytrain' = [an N x 1 vector of observations of your target],
					'dates_ins' = [OPTIONAL: a vector of dates corresp[onding to your insample observations; if set to NULL the 'date' columns in the output matrices will be numbered as 1,...,nrow('Xtrain')],
					'model_object' = NULL,
					'lmbda' = [the L2-type penalty parameter]
		  )



% ---------------------------- Type: NN ---------------------------- %


run__model <- list(
			'type' = 'NN',
			'params' = list(
					'Xtrain' = NULL,
					'Xtest' = NULL,
					'Ytrain' = [an N x 1 vector of observations of your target],
					'dates_ins' = [OPTIONAL: a vector of dates corresp[onding to your insample observations; if set to NULL the 'date' columns in the output matrices will be numbered as 1,...,nrow('Xtrain')],
					'model_object' = [an MLP-type object, i.e. a neural network that was fitted using the provided MLP.R function],
					'lmbda' = NULL
		  )


% ======================= Output of the Function ====================== #


 - weights:		an N x T matrix of observation weights, where N denotes the number of insample observations, and T the number of out-of-sample forecast dates

 - contributions: 	an N x T matrix of observation contributions, where N denotes the number of insample observations, and T the number of out-of-sample forecast dates

 - concentration:	Forecast Concentration, as described in the paper

 - short_position:	Forecast Short Position, as described in the paper

 - turnover:		Forecast Turnover, as described in the paper

 - leverage:		Forecast Leverage, as described in the paper	
