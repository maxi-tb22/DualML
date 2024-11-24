#################################################################################################################
#
#
#                       Goulet Coulombe, Philippe, and Goebel, Maximilian, and Klieber, Karin
#
#                             "Dual Interpretation of Machine Learning Forecasts"
#
#################################################################################################################

# --- This version: 2024-11-17



DualML <- function(run__model,run__type='regression',Q=5){
  
  
  # --- Q: used for calculating Forecast Concentration and denotes the proportion of total weights attributed to the top Q% of the weights
  
  
  if (run__type != 'regression'){
    stop(sprintf('\nSorry, DualML is currently only implemented for regression-type models. \n'))
  }
  

  

  # ----------------------------------------   00. Plain OLS-type models ---------------------------------------- #
  
  if (run__model[['type']] == 'OLS'){
    
    
    # ------------------------------------- Observation Weights ------------------------- #
    
    # --- Projection-Matrix
    project_matrix <- solve(t(run__model[['params']][['Xtrain']]) %*% as.matrix(run__model[['params']][['Xtrain']]))
    # --- OOS Observation Weights
    df_weights <- setNames(data.frame(t(as.matrix(run__model[['params']][['Xtest']]) %*% project_matrix %*% t(run__model[['params']][['Xtrain']]))),
                             nm=paste0('obs_oos_',c(1:nrow(run__model[['params']][['Xtest']]))))
    
    
    
    # ------------------------------------- Observation Contributions ------------------------- #
    
    # --- OOS Observation Contributions
    df_contributions <- sapply(c(1:ncol(df_weights)),function(x) cumsum(df_weights[,x] * (run__model[['params']][['Ytrain']] - mean(run__model[['params']][['Ytrain']]))) + mean(run__model[['params']][['Ytrain']]))
    
    # --- Make it prettier:
    df_contributions <- setNames(data.frame(df_contributions), nm=colnames(df_weights))
    
    
    # ------------------------------------- Attach the Observation Label ------------------------------------- #
    if (is.null(run__model[['params']][['dates_ins']])){
      df_weights[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
      df_contributions[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
    } else {
      df_weights[,'date'] <- run__model[['params']][['dates_ins']]
      df_contributions[,'date'] <- run__model[['params']][['dates_ins']]
    }
    
    
  }
  
  # ----------------------------------------   1. Random Forest ---------------------------------------- #
  
  if (run__model[['type']] == 'RF'){
    
    
    
    # ------------------------------------- Observation Weights ------------------------- #
    
    # --- OOS Observation Weights
    df_weights <- as.data.frame(as.matrix(rf_weights(run__model[['params']][['model_object']], 
                                                     run__model[['params']][['Ytrain']], 
                                                     run__model[['params']][['Xtrain']], 
                                                     run__model[['params']][['Xtest']]),
                                          dimnames = list(c(),paste0('obs_oos_',c(1:nrow(run__model[['params']][['Xtest']]))))))
    
    
    
    # ------------------------------------- Observation Contributions ------------------------- #
    
    # --- OOS Observation Contributions
    df_contributions <- sapply(c(1:ncol(df_weights)),function(x) cumsum(df_weights[,x] * (run__model[['params']][['Ytrain']] - mean(run__model[['params']][['Ytrain']]))) + mean(run__model[['params']][['Ytrain']]))
    
    # --- Make it prettier:
    df_contributions <- setNames(data.frame(df_contributions), nm=colnames(df_weights))
    
    
    # ------------------------------------- Attach the Observation Label ------------------------------------- #
    if (is.null(run__model[['params']][['dates_ins']])){
      df_weights[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
      df_contributions[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
    } else {
      df_weights[,'date'] <- run__model[['params']][['dates_ins']]
      df_contributions[,'date'] <- run__model[['params']][['dates_ins']]
    }
    
    
  }
  
  
  # ----------------------------------------   2. Boosted Trees (LightGBM) ---------------------------------------- #
  
  if (run__model[['type']] == 'LGB'){
    
    
    
    # ------------------------------------- Observation Weights ------------------------- #
    
    # --- OOS Observation Weights
    df_weights <- GeertsemaLu2023(run__model[['params']][['model_object']], 
                                  run__model[['params']][['Ytrain']], 
                                  run__model[['params']][['Xtrain']], 
                                  run__model[['params']][['Xtest']])
    df_weights <- setNames(data.frame(df_weights), nm=paste0('obs_oos_',c(1:nrow(run__model[['params']][['Xtest']]))))
    
    
    
    # ------------------------------------- Observation Contributions ------------------------- #
    
    # --- OOS Observation Contributions
    df_contributions <- sapply(c(1:ncol(df_weights)),function(x) cumsum(df_weights[,x] * (run__model[['params']][['Ytrain']] - mean(run__model[['params']][['Ytrain']]))) + mean(run__model[['params']][['Ytrain']]))
    
    # --- Make it prettier:
    df_contributions <- setNames(data.frame(df_contributions), nm=colnames(df_weights))
    
    
    # ------------------------------------- Attach the Observation Label ------------------------------------- #
    if (is.null(run__model[['params']][['dates_ins']])){
      df_weights[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
      df_contributions[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
    } else {
      df_weights[,'date'] <- run__model[['params']][['dates_ins']]
      df_contributions[,'date'] <- run__model[['params']][['dates_ins']]
    }
    
    
  }
  
  
  # ----------------------------------------   3. Kernel Ridge-Regression ---------------------------------------- #
  
  if (run__model[['type']] == 'KRR'){
    
    
    
    # ------------------------------------- Observation Weights ------------------------- #
    
    # --- Penalty-Matrix
    pen_mat <- run__model[['params']][['lmbda']] * diag(nrow(run__model[['params']][['Xtrain']]))
    # --- OOS Observation Weights
    df_weights <- setNames(data.frame(t(run__model[['params']][['Xtest']] %*% solve(run__model[['params']][['Xtrain']] +  pen_mat))),
                           nm=paste0('obs_oos_',c(1:nrow(run__model[['params']][['Xtest']]))))
    
    
    
    # ------------------------------------- Observation Contributions ------------------------- #
    
    # --- Sum of Weights:
    KRR_sum_weights <- apply(df_weights,2,sum)
    
    # --- OOS Observation Contributions
    df_contributions <- sapply(c(1:ncol(df_weights)),function(x) cumsum(df_weights[,x] * (run__model[['params']][['Ytrain']] - mean(run__model[['params']][['Ytrain']]))) + mean(run__model[['params']][['Ytrain']]) * KRR_sum_weights[x])
    
    # --- Make it prettier:
    df_contributions <- setNames(data.frame(df_contributions), nm=colnames(df_weights))
    
    
    # ------------------------------------- Attach the Observation Label ------------------------------------- #
    if (is.null(run__model[['params']][['dates_ins']])){
      df_weights[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
      df_contributions[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
    } else {
      df_weights[,'date'] <- run__model[['params']][['dates_ins']]
      df_contributions[,'date'] <- run__model[['params']][['dates_ins']]
    }
    
    
    
  }
  
  
  # ----------------------------------------   4. Neural Network (PGC) ---------------------------------------- #
  
  if (run__model[['type']] == 'NN'){
    
    if (!('portfolio.weights' %in% names(run__model[['params']][['model_object']]))){
      stop(sprintf('\nI cannot find the \'portfolio.weights\' in your \'model_object\'. 
                   \nAre you sure your NN was fitted using the provided MLP function?. \n'))
    } else {
      
      # --- Some necessary SHADOW-matrices for compatibility:
      run__model[['params']][['Xtrain']] <- data.frame(matrix(NA,nrow=length(run__model[['params']][['Ytrain']]),ncol=1))
      run__model[['params']][['Xtest']] <- data.frame(matrix(NA,nrow=length(run__model[['params']][['model_object']][['pred']]),ncol=1))
      
    }
    
    # ------------------------------------- Observation Weights ------------------------- #
    
    # --- OOS Observation Weights
    df_weights <- setNames(data.frame(t(run__model[['params']][['model_object']][['portfolio.weights']])),
                           nm=paste0('obs_oos_',c(1:nrow(run__model[['params']][['model_object']][['portfolio.weights']]))))
    
    
    
    # ------------------------------------- Observation Contributions ------------------------- #
    
    
    # --- OOS Observation Contributions
    df_contributions <- sapply(c(1:ncol(df_weights)),function(x) cumsum(df_weights[,x] * ((run__model[['params']][['Ytrain']] - mean(run__model[['params']][['Ytrain']]))/sd(run__model[['params']][['Ytrain']]))) * sd(run__model[['params']][['Ytrain']]) + mean(run__model[['params']][['Ytrain']]))
    
    # --- Make it prettier:
    df_contributions <- setNames(data.frame(df_contributions), nm=colnames(df_weights))
    
    
    # ------------------------------------- Attach the Observation Label ------------------------------------- #
    if (is.null(run__model[['params']][['dates_ins']])){
      df_weights[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
      df_contributions[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
    } else {
      df_weights[,'date'] <- run__model[['params']][['dates_ins']]
      df_contributions[,'date'] <- run__model[['params']][['dates_ins']]
    }
    
  }
  
  
  # ----------------------------------------   5. Ridge Regression ---------------------------------------- #
  
  if (run__model[['type']] == 'RR'){
    
    
    # ------------------------------------- Observation Weights ------------------------- #
    
    # --- Projection-Matrix
    project_matrix <- solve(t(run__model[['params']][['Xtrain']]) %*% as.matrix(run__model[['params']][['Xtrain']]) + nrow(run__model[['params']][['Xtrain']])*run__model[['params']][['lmbda']]*diag(1,ncol(run__model[['params']][['Xtrain']])))
    # --- OOS Observation Weights
    df_weights <- setNames(data.frame(t(as.matrix(run__model[['params']][['Xtest']]) %*% project_matrix %*% t(run__model[['params']][['Xtrain']]))),
                           nm=paste0('obs_oos_',c(1:nrow(run__model[['params']][['Xtest']]))))
    
    
    
    # ------------------------------------- Observation Contributions ------------------------- #
    
    
    # --- OOS Observation Contributions
    df_contributions <- sapply(c(1:ncol(df_weights)),function(x) cumsum(df_weights[,x] * (run__model[['params']][['Ytrain']] - mean(run__model[['params']][['Ytrain']]))) + mean(run__model[['params']][['Ytrain']]))
    
    # --- Make it prettier:
    df_contributions <- setNames(data.frame(df_contributions), nm=colnames(df_weights))
    
    
    # ------------------------------------- Attach the Observation Label ------------------------------------- #
    if (is.null(run__model[['params']][['dates_ins']])){
      df_weights[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
      df_contributions[,'date'] <- c(1:nrow(run__model[['params']][['Xtrain']]))
    } else {
      df_weights[,'date'] <- run__model[['params']][['dates_ins']]
      df_contributions[,'date'] <- run__model[['params']][['dates_ins']]
    }
    
  }
  
  
  
  
  
  # ------------------------------------ Forecast Concentration ------------------------------------ #
  k <- dim(df_weights)[1]*Q/100
  metric_FC <- round(FC_CR(df_weights[,1:nrow(run__model[['params']][['Xtest']])],k),4)
  
  # ------------------------------------ Forecast Short Position ------------------------------------ #
  metric_FSP <- round(FSP(df_weights[,1:nrow(run__model[['params']][['Xtest']])]),4)
  
  # ------------------------------------ Forecast Turnover ------------------------------------ #
  metric_FT <- round(FT(df_weights[,1:nrow(run__model[['params']][['Xtest']])]),4)
  
  # ------------------------------------ Forecast Leverage ------------------------------------ #
  metric_FL <- round(as.vector(sapply(c(1:nrow(run__model[['params']][['Xtest']])), function(x) sum(df_weights[,x]))),2)
  
  if ((run__model[['type']] == 'RR') | (run__model[['type']] == 'OLS')){
    if (run__model[['params']][['intercept']] == F){
      metric_FL <- metric_FL + 1
    }
  }
  
  
  # ------------------------------------ Export Results ------------------------------------ #
  return(list('weights'=df_weights,
              'contributions'=df_contributions,
              'concentration'=metric_FC,
              'short_position'=metric_FSP,
              'turnover'=metric_FT,
              'leverage'=metric_FL))

}






# ======================================== Geertsema & Lu (2023) ======================================== #


GeertsemaLu2023 <- function(mod,Y_train,X_train,X_test,show.progress = F){
  
  # --------------------------------- Leaf Coincidence Matrix --------------------------------- #
  LCM <- function(leaf_vec_1,leaf_vec_2){
    
    mat <- matrix(NA, nrow=length(leaf_vec_1), ncol = length(leaf_vec_2))
    
    for (rr in 1:length(leaf_vec_1)){
      for (cc in 1:length(leaf_vec_2)){
        if (leaf_vec_1[rr] == leaf_vec_2[cc]){
          mat[rr,cc] <- 1
        } else {
          mat[rr,cc] <- 0
        }
      }
      
    }
    
    return(mat)
    
  }
  
  # ------------------------------------- Algo I ------------------------------------- #
  
  # --- Number of trees &  Learning Rate
  if ('best_ntreelimit' %in% names(mod)){
    N_trees <- mod$best_ntreelimit
    learning_rate <- mod[['params']][['eta']]
    mod_type <- 'xgb'
  } else if ('best_iter' %in% names(mod)){
    N_trees <- mod$best_iter
    learning_rate <- mod[['params']][['learning_rate']]
    mod_type <- 'lgb'
  } else {
    stop('ERROR: Cannot identify the Number of Trees!')
  }
  
  
  
  # --- Number of training samples
  N_train = length(Y_train)
  
  
  # --- Storage for the "Tree-Prediction-Weight-Matrix"
  P_list <- list()
  
  # --- Initialize the prediction matrix
  #if (objective == 'regression'){
  #  base_score <- 1/N_train
  #} else if (objective == 'binary_classification'){
  #  base_score <- log_odds(0.5)
  #}
  G <- 1/N_train * matrix(1,nrow=N_train,ncol=N_train)
  
  P_list[[1]] <- G
  
  if (show.progress) {
    pb <- txtProgressBar(min = 0, max = N_trees+N_trees, style = 3) # Progress bar 
  }
  # --- Run through the trees!
  for (tt in 1:N_trees){
    
    
    # --- For tree 'tt', which observations have been assigned to which leaf in-sample?
    if (mod_type == 'xgb'){
      v_tt <- predict(mod, xgb.DMatrix(data=as.matrix(X_train)), predleaf = TRUE)[,tt]
    } else if (mod_type == 'lgb'){
      v_tt <- predict(mod, as.matrix(X_train), params=list(leaf_index = TRUE))[,tt]
    }
    
    
    # --- Construct Leave-Coincidence Matrix
    D <- LCM(v_tt,v_tt)
    
    # --- Scale the Leave Coincidence Matrix
    W <- D / (matrix(1,nrow=N_train,ncol=N_train) %*% D)
    
    # --- Update the Tree-Prediction-Weight-Matrix
    P <- learning_rate * W %*% (eye(N_train) - G)
    
    # --- Update the Prediction-Matrix for next iteration
    G <- G + P
    
    # --- Collect: Tree-Prediction-Weight-Matrix
    P_list[[tt+1]] <- P
    
    
    # --- Next 'tt'
    if (show.progress) {
      setTxtProgressBar(pb, tt) 
    }
  }
  
  
  
  
  
  # ------------------------------------- Algo II ------------------------------------- #
  
  # --- Number of Test observations
  N_oos <- nrow(X_test)
  
  # --- Initialize the "Tree-Prediction-Weight-Matrix"
  P <- P_list[[1]]
  
  # --- Initialize 'L'
  L <- matrix(1,nrow=N_train,ncol=N_oos) 
  
  # --- First-iteration prediction weights
  K <- t(P) %*% (L / (matrix(1,nrow=N_train,ncol=N_train) %*% L))
  
  # --- Run through the trees!
  for (tt in 1:N_trees){
    
    # --- For tree 'tt', which observations have been assigned to which leaf in-sample?
    if (mod_type == 'xgb'){
      v_tt <- predict(mod, xgb.DMatrix(data=as.matrix(X_train)), predleaf = TRUE)[,tt]
    } else if (mod_type == 'lgb'){
      v_tt <- predict(mod, as.matrix(X_train), params=list(leaf_index = TRUE))[,tt]
    }
    
    # --- For tree 'tt', which observations have been assigned to which leaf out-of-sample?
    if (mod_type == 'xgb'){
      w_tt <- predict(mod, xgb.DMatrix(data=as.matrix(X_test)), predleaf = TRUE)[,tt]
    } else if (mod_type == 'lgb'){
      w_tt <- predict(mod, as.matrix(X_test), params=list(leaf_index = TRUE))[,tt]
    }
    
    # --- Get the "Tree-Prediction-Weight-Matrix"
    P <- P_list[[tt+1]]
    
    # --- Construct Leave-Coincidence Matrix
    L <- LCM(v_tt,w_tt)
    
    # --- Scale the Leaf-Coincidence-Matrix
    W <- L / (matrix(1,nrow=N_train,ncol=N_train) %*% L)
    
    # --- Update/Accumulate the Prediction-weights
    K <- K + (t(P) %*% W)
    
    
    
    
    # --- Next 'tt'
    if (show.progress) {
      setTxtProgressBar(pb, N_trees + tt) 
    }
  }
  
  
  
  return(K)
  
  
}



# ======================================== RF Weights ======================================== #


rf_weights <- function(mod_RF, Ytrain, Xtrain, Xtest, show.progress=FALSE){
  
  
  # """ A function to get the weights of each observation in the presence of out-of-bag samples! """ 
  
  
  # --- output:     length(Ytrain) x length(Ytest) matrix of observation weights
  # --- --- remark: out-of-bag observations are accounted for, i.e. ELIMINATED!
  
  if (is.null(mod_RF$inbag.counts)){
    stop('ERROR: Set \'keep.inbag=TRUE\' in the ranger function!')
  }
  
  # --- Which observations have been assigned to which leaf in-sample?
  rf_assigned_leaf_ins <- predict(mod_RF, Xtrain, type = 'terminalNodes', predict.all = TRUE)$predictions
  
  # --- Which observations have been assigned to which leaf out-of-sample?
  rf_assigned_leaf_oos <- predict(mod_RF, Xtest, type = 'terminalNodes', predict.all = TRUE)$predictions
  
  
  # --- Get the prediction from each tree
  obs_weights_by_oos <- as.data.frame(matrix(0,nrow=length(Ytrain),ncol=nrow(Xtest),
                                             dimnames = list(c(),paste0('obs_oos_',c(1:nrow(Xtest))))))
  
  if (show.progress) {
    pb <- txtProgressBar(min = 0, max = ncol(obs_weights_by_oos), style = 3) # Progress bar 
  }
  for (oo in 1:ncol(obs_weights_by_oos)){
    
    # --- Get predictions by Tree:
    obs_weights_by_tree <- as.data.frame(matrix(0,nrow=length(Ytrain),ncol=length(mod_RF$inbag.counts),
                                                dimnames = list(c(),paste0('Tree_',c(1:length(mod_RF$inbag.counts))))))
    
    # --- Run over all trees for OOS-period 'oo'
    for (ll in 1:length(mod_RF$inbag.counts)){
      
      # --- Get the leaf that the oos-prediction came out of
      leaf_ll <- rf_assigned_leaf_oos[oo,ll]
      
      # --- Get the in-sample indices that are assigned to 'leaf_ll' and that are NOT OOB:
      ins_idx <- which((rf_assigned_leaf_ins[,ll] == leaf_ll) & (mod_RF$inbag.counts[[ll]]) == 1)
      
      if (length(ins_idx) > 0){
        obs_weights_by_tree[ins_idx, ll] <- 1/length(ins_idx)
      }
      
    }
    
    # --- Average across across trees:
    obs_weights_by_oos[, oo] <- apply(obs_weights_by_tree,1,mean)
    
    if (show.progress) {
      setTxtProgressBar(pb, oo) 
    }
  }
  
  
  # --- Return the weights
  return(obs_weights_by_oos)
  
}



# ======================================== Metrics ======================================== #



FC_CR <- function(mat,k){
  # --- Forecast Concentration
  
  # --- mat:  a T_ins by T_oos matrix of observation weights, 
  #           where T_oos is the number of out-of-sample observations, i.e. forecasts
  
  if (is.null(dim(mat))){
    mat <- as.matrix(mat)
  }
  result <- apply(mat, 2, function(x) {
    # Sort the vector in decreasing order and sum the top k elements
    top_k_sum <- sum(sort(abs(x), decreasing = TRUE)[1:k])
    # Calculate the total sum of the vector
    total_sum <- sum(abs(x))
    # Return the ratio
    return(top_k_sum / total_sum)
  })
}


FSP <- function(mat){
  
  # --- Forecast Short-Position
  
  # --- mat:  a T_ins by T_oos matrix of observation weights, 
  #           where T_oos is the number of out-of-sample observations, i.e. forecasts
  
  if (is.null(dim(mat))){
    mat <- as.matrix(mat)
  }
  return(apply(mat,2,function(x) sum(x*I(x < 0))))
}


FT <- function(mat){
  
  # --- Forecast Turnover
  
  # --- mat:  a T_ins by T_oos matrix of observation weights, 
  #           where T_oos is the number of out-of-sample observations, i.e. forecasts
  
  if (is.null(dim(mat))){
    return(NA)
  } else {
    # --- Create a first-difference-matrix by column:
    mat_fd <- c()
    for (cc in 2:ncol(mat)){
      mat_fd <- cbind(mat_fd,abs(mat[,cc]-mat[,cc-1]))
    }
    
    # --- Sum across columns, then across rows
    return(sum(apply(mat_fd,1,sum)))
  }
  
  
}
