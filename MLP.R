MLP <- function(X,Y,Xtest,Ytest,X_OOS=as.matrix(),nn_hyps,standardize,seed) {
  
  if (require(torch, quietly=T) == F){
    stop(sprintf('\nTo run the \'MLP.R\' function, you need to install \'torch\'! '))
  }
  
  
  if (is.null(nn_hyps$objective)){
    nn_hyps$objective <- 'regression'
  }
  if (is.null(nn_hyps$optim_KLR)){
    nn_hyps$optim_KLR <- 'adam'
  }
  if (is.null(nn_hyps$keep_all)) {
    nn_hyps$keep_all <- TRUE
  }
  
  # --- For compatibility! Gosh this is such a pain...
  if (nn_hyps$objective == 'binary_classification'){
    nn_hyps[['lambda_grid']] <- nn_hyps[['lambda_grid_SIGMOID']]
  }
  
  set.seed(seed)
  
  show_train=nn_hyps$show_train
  
  if(show_train < 3) {
    
    cat("\nProgress : \n")
    cat(rep("-",40), sep = "")
    cat("\n \n")
    
  }
  
  # If needed scale data
  temp <- c()
  
  if(standardize==T) {
    
    if(show_train < 3) {
      cat("Standardize Data !!!! \n \n")
    }
    
    temp=scale_data(Xtrain = X, Ytrain = Y, Xtest = Xtest, Ytest = Ytest)
    X=temp$Xtrain
    Xtest=temp$Xtest
    
    if (nn_hyps$objective == 'binary_classification'){
      Ytest = Ytest
      Y = Ytrain
    } else if (nn_hyps$objective == 'regression'){
      Ytest = temp$Ytest
      Y = temp$Ytrain
    }
  }
  
  # Convert our input data and labels into tensors.
  x_train = torch_tensor(X, dtype = torch_float())
  x_test = torch_tensor(Xtest, dtype = torch_float())
  x_OOS = torch_tensor(X_OOS, dtype = torch_float())
  y_train = torch_tensor(Y, dtype = torch_float())
  y_test = torch_tensor(Ytest, dtype = torch_float())
  
  
  
  
  # =====================================================================================================
  ## MNN MODEL
  # =====================================================================================================
  
  if(show_train < 3) {
    cat("Initialize Model !!!!! \n \n") 
  }
  
  BuildNN <- function(X,Y,training_index,nn_hyps) {
    
    # Setting up hyperparameters
    lr=nn_hyps$lr
    epochs=nn_hyps$epochs
    patience=nn_hyps$patience
    tol=nn_hyps$tol
    
    #training_index=training_index  
    show_train=nn_hyps$show_train
    
    # Build Model
    net = nn_module(
      "nnet",
      
      initialize = function(n_features=nn_hyps$n_features, nodes=nn_hyps$nodes,dropout_rate=nn_hyps$dropout_rate){
        
        self$n_layers = length(nodes)
        
        self$input = nn_linear(n_features,nodes[1])
        
        if(length(nodes)==1) {
          self$first = nn_linear(nodes,nodes)
        }else{
          self$first = nn_linear(nodes[1],nodes[1])
          self$hidden <- nn_module_list(lapply(1:(length(nodes)-1), function(x) nn_linear(nodes[x], nodes[x+1])))
        }
        
        self$output = nn_linear(nodes[length(nodes)],1)
        self$dropout = nn_dropout(p=dropout_rate)
        
      },
      
      forward = function(x){
        
        # Input
        x = torch_relu(self$input(x))
        
        # Hidden
        x = torch_relu(self$first(x))
        x = self$dropout(x)
        
        if (nn_hyps$objective == 'binary_classification'){
          if(self$n_layers>1) {
            
            for(layer in 1:(self$n_layers-2)) {
              x = torch_relu(self$hidden[[layer]](x))
              x = self$dropout(x)
            }
            # --- Last layer before output shall be linear, otherwise the sigmoid will generate a lower bound of 0.5
            x = self$hidden[[self$n_layers-1]](x)
            x = self$dropout(x)
            
          }
          
        } else if (nn_hyps$objective == 'regression'){
          if(self$n_layers>1) {
            
            for(layer in 1:(self$n_layers-1)) {
              x = torch_relu(self$hidden[[layer]](x))
              x = self$dropout(x)
            }
            
          }
        }
        
        
        # Output
        if (nn_hyps$objective == 'binary_classification'){
          yhat = torch_squeeze(torch_sigmoid(self$output(x)))
        } else if (nn_hyps$objective == 'regression'){
          yhat = torch_squeeze(self$output(x))
        }
        
        result <- list(yhat,x)
        return(result)
      }
      
    )
    
    model = net()
    
    
    ## Train model ---------------------------------------------------------------
    # ----------------------------------------------------------------------------
    
    patience = patience
    wait = 0
    
    oob_index <- c(1:x_train$size()[1])[-training_index]
    
    batch_size <- nn_hyps$batch_size
    num_data_points <- length(training_index)
    num_batches <- floor(num_data_points/batch_size)
    
    if(!is.na(nn_hyps$num_batches)){
      num_batches <- nn_hyps$num_batches
    }
    
    best_epoch = 0
    best_loss = NA
    
    if (nn_hyps$objective == 'binary_classification'){
      criterion = nn_bce_loss()
    } else if (nn_hyps$objective == 'regression'){
      criterion = nn_mse_loss()
    }
    
    optimizer = optim_adam(model$parameters, lr = lr)
    
    
    for (i in 1:epochs) {
      
      # manually loop through the batches
      training_index <- sample(training_index)
      model$train()
      for(batch_idx in 1:num_batches) {
        
        optimizer$zero_grad() # Start by setting the gradients to zero
        
        # here index is a vector of the indices in the batch
        idx <- (batch_size*(batch_idx-1) + 1):(batch_idx*batch_size)
        
        # train
        y_pred=model(x_train[training_index[idx],])[[1]]
        loss=criterion(y_pred,y_train[training_index[idx]])
        
        loss$backward()  # Backpropagation step
        optimizer$step() # Update the parameters
        
        # Check Training
        if(show_train==1) {
          if(i %% 1 == 0) {
            #cat(" Batch number: ",batch_idx," on ", num_batches, "\n")
            # cat(" Epoch:", i, "Loss: ", round(loss$item(),5),", Val Loss: ",round(loss_oob$item(),5), "\n")
            
          }
        }
        
      }
      
      model$eval()
      with_no_grad({
        y_pred_oob=model(x_train[oob_index,])[[1]]
        loss_oob=criterion(y_pred_oob,y_train[oob_index])
        
        if(x_train$size()[1] == length(training_index)) {
          loss_oob=criterion(y_pred,y_train[training_index])
        }
      })
      
      percentChange <- ((best_loss - loss_oob$item())/loss_oob$item())
      
      # Early Stopping
      if(best_loss > loss_oob$item() | i == 1) { #best_loss > loss_oob$item()
        best_loss=loss_oob$item()
        best_epoch=i
        state_best_model <- lapply(model$state_dict(), function(x) x$clone()) 
        
        if(percentChange > tol | i == 1) {
          wait=0
        }else {
          wait=wait+1
        }
        
      }else{
        
        wait=wait+1
        
      }
      
      if(show_train==1) {
        
        # Check Training
        if(i %% 1 == 0) {
          cat(" Epoch:", i, ", Loss: ", loss$item(),", Val Loss: ",loss_oob$item(), "(PercentChange: ",round(percentChange,3),")", "\n")
          # cat(" Epoch:", i, "Loss: ", round(loss$item(),5),", Val Loss: ",round(loss_oob$item(),5), "\n")
          
        }
        
      }
      
      if(wait > patience) {
        if(show_train==1) {
          cat("Best Epoch at:", best_epoch, "\n")
        }
        break
      }
      
    }
    
    model$load_state_dict(state_best_model)
    return(model) # Return the model with the best val loss
    
  }
  
  # =====================================================================================================
  ## MODEL AVERAGING
  # =====================================================================================================
  
  num_average <- nn_hyps$num_average
  sampling_rate <- nn_hyps$sampling_rate
  pred.in.ensemble <- array(data = NA, dim = c(nrow(X),num_average))
  pred.ensemble <- array(data = NA, dim = c(nrow(Xtest),num_average))
  pred.OOS.ensemble <- array(data = NA, dim = c(nrow(X_OOS),num_average))
  inner.prod = array(data = NA, dim = c(nrow(Xtest),nrow(X),num_average))
  lambda_opt <- rep(NA, num_average)
  mse_opt <- rep(NA, num_average)
  
  
  if (nn_hyps$objective == 'binary_classification'){
    NNkbc_klr_LOGODDS_contrib_shit = array(data = NA, dim = c(nrow(Xtest)+nrow(X_OOS),nrow(X),num_average))
    NNkbc_klr_PROBA_contrib_shit = array(data = NA, dim = c(nrow(Xtest)+nrow(X_OOS),nrow(X),num_average))
    
     pred.in.k_klr <- array(data = NA, dim = c(nrow(X),num_average))
    
    pred.k_klr <- array(data = NA, dim = c(nrow(Xtest),num_average))
    
    pred.OOS.k_klr <- array(data = NA, dim = c(nrow(X_OOS),num_average))
    
    lambda_opt_KLR <- rep(NA, num_average)
    mse_opt_KLR <- rep(NA, num_average)
    
    mse_ins_KLR <- array(NA,dim=c(length(nn_hyps[['lambda_grid_KLR']]),num_average))
    mse_oos_KLR <-  array(NA,dim=c(length(nn_hyps[['lambda_grid_KLR']]),num_average))
    
    conv_list <- list()
    log_loss_list <- list()
    
  } else if (nn_hyps$objective == 'regression'){
    NNkbc_contrib_shit = array(data = NA, dim = c(nrow(Xtest)+nrow(X_OOS),nrow(X),num_average))
    pred.in.k <- array(data = NA, dim = c(nrow(X),num_average))
    pred.k <- array(data = NA, dim = c(nrow(Xtest),num_average))
    pred.OOS.k <- array(data = NA, dim = c(nrow(X_OOS),num_average))
  }
  
  
  
  if(show_train==2) {
    pb <- txtProgressBar(min = 0, max = num_average, style = 3) # Progress bar
  }
  
  trained_model <- list()
  for(j in 1:num_average) {
    
    # Bootstrap parameters
    set.seed(seed+j)
    boot <- sample(1:nrow(X), size = sampling_rate*nrow(X), replace = F) # training
    oob <- (nrow(X)+1):(nrow(X)+nrow(Xtest))                             # out of bag
    
    ## Estimation -------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    model <- BuildNN(x_train,y_train,boot,nn_hyps)
    trained_model[[j]] <- model
    
    if(show_train==1) {
      cat("Done with model :",j, "\n \n")
    }
    
    ## Storage ----------------------------------------------------------------
    #--------------------------------------------------------------------------
    
    model$eval()
    
    if(standardize==T) {
      
      if (nn_hyps$objective == 'binary_classification'){
        pred.in.ensemble[,j] <- as.matrix(model(x_train[,])[[1]])
        pred.ensemble[,j] <- as.matrix(model(x_test)[[1]])
        if (nrow(X_OOS) > 0){
          pred.OOS.ensemble[,j] <- as.matrix(model(x_OOS)[[1]])
        }
      } else if (nn_hyps$objective == 'regression'){
        pred.in.ensemble[,j] <- invert_scaling(as.matrix(model(x_train[,])[[1]]),temp)
        pred.ensemble[,j] <- invert_scaling(as.matrix(model(x_test)[[1]]),temp)
        if (nrow(X_OOS) > 0){
          pred.OOS.ensemble[,j] <- invert_scaling(as.matrix(model(x_OOS)[[1]]),temp)
        }
      }
      
       
      
      # Convert the model outputs to matrices
      train_embeddings <- as.matrix(model(x_train)[[2]])
      test_embeddings <- as.matrix(model(x_test)[[2]])
      if (nrow(X_OOS) > 0){
        oos_embeddings <- as.matrix(model(x_OOS)[[2]])
      }
      
      # Compute the cross-product of the training embeddings
      train_crossprod <- tcrossprod(train_embeddings)
      
      # Compute the kernel matrix
      K.out <- tcrossprod(test_embeddings,train_embeddings)
      if (nrow(X_OOS) > 0){
        K.oos <- tcrossprod(oos_embeddings,train_embeddings)
      }
      
      # --- Find the DUAL Solution:
      if (nn_hyps$objective == 'binary_classification'){
        
         
        # -------------------------------- Prediction via Kernel-Logistic-Regression ------------------------------- #
        
        # --- Run the Grid-Search SHIT! This is pure pain!
        mse_lambda <- mse_oos_lambda <- c()
        conv_list_j <- log_loss_list_j <- list()
        count <- 0
        for (la in nn_hyps[['lambda_grid_KLR']]){
          count <- count+1
          # --- Fitting:
          out_KLR <- kernel_logistic_regression(train_embeddings,Y,
                                                kernel_params=list(type='linear'),
                                                lambda = la, 
                                                optimizer = nn_hyps[['optim_KLR']],
                                                maxit = nn_hyps[['maxit_KLR']], lr = nn_hyps[['lr_KLR']], eps=1e-15,
                                                verbose=FALSE)
          
          conv_list_j[[count]] <- out_KLR$fi
          log_loss_list_j[[count]] <- out_KLR$logloss
          
          # --- Prediction: In-Sample
          yhat_ins_klr <- predict_klr(out_KLR,train_embeddings, train_embeddings)$predictions
          
          mse_lambda <- c(mse_lambda, mean((pred.in.ensemble[,j] - yhat_ins_klr)^2)/var(pred.in.ensemble[,j]))
          
          # --- Prediction: Out-Of-Sample
          yhat_oos_klr <- predict_klr(out_KLR,train_embeddings, test_embeddings)$predictions
          
          mse_oos_lambda <- c(mse_oos_lambda, mean((pred.ensemble[,j] - yhat_oos_klr)^2)/var(pred.ensemble[,j]))
          
          
        }
        
        # --- Which was the optimal lambda?
        mse_opt_KLR[j] <- min(mse_lambda)
        lambda_opt_KLR[j] <- nn_hyps[['lambda_grid_KLR']][which(mse_lambda == min(mse_lambda))][1]
        
        mse_ins_KLR[,j] <- mse_lambda
        mse_oos_KLR[,j] <- mse_oos_lambda
        
        # --- Fitting, using the 'optimal' lambda
        out_KLR <- kernel_logistic_regression(train_embeddings,Y,
                                              kernel_params=list(type='linear'),
                                              lambda = lambda_opt_KLR[j], 
                                              optimizer = nn_hyps[['optim_KLR']],
                                              maxit = nn_hyps[['maxit_KLR']], lr = nn_hyps[['lr_KLR']], eps=1e-15,
                                              verbose=TRUE)
        
        
        
        if (out_KLR$algo_conv | nn_hyps[['keep_all']]) { #out_KLR$algo_conv
          
          conv_list_j[[count+1]] <- out_KLR$fi
          log_loss_list_j[[count+1]] <- out_KLR$logloss
          
          names(conv_list_j) <- names(log_loss_list_j) <- c(paste0('la',nn_hyps[['lambda_grid_KLR']]),'final')
          conv_list[[j]] <- conv_list_j
          log_loss_list[[j]] <- log_loss_list_j
          
          par(mfrow=c(3,2))
          ts.plot(na.omit(out_KLR$fi[-c(1:10)]), main = paste0("convergence, la",lambda_opt_KLR[j]), ylab="",xlab="")
          ts.plot(na.omit(log(out_KLR$fi)), main="log convergence", ylab="",xlab="")
          ts.plot(na.omit(out_KLR$logloss[-c(1:10)]), main="logloss", ylab="",xlab="")
          ts.plot(na.omit(log(out_KLR$logloss[-c(1:10)])), main="loglogloss", ylab="",xlab="")
          #ts.plot(mse_lambda, main="MSE INS", ylab = "", xlab="lambda")
          #ts.plot(mse_oos_lambda, main="MSE OOS", ylab = "", xlab="lambda")
          plot(x = as.factor(nn_hyps[['lambda_grid_KLR']]), y = mse_lambda, main="MSE INS", ylab = "", xlab = "lambda")
          plot(x = as.factor(nn_hyps[['lambda_grid_KLR']]), y = mse_oos_lambda, main="MSE OOS", ylab = "", xlab = "lambda")
          
          # --------------------- Collect the results ----------------------- #
          
          
          # --- Training Set
          pred.in.k_klr[,j] = predict_klr(out_KLR,train_embeddings, train_embeddings)$predictions
          
          # --- Test Set
          pred.k_klr[,j] = predict_klr(out_KLR,train_embeddings, test_embeddings)$predictions
          
          # --- OOS
          if (nrow(X_OOS) > 0){
            pred.OOS.k_klr[,j] = predict_klr(out_KLR,train_embeddings, oos_embeddings)$predictions
          }
          
          
          # --- Non solo le maledette previste ma anche i contribuzioni! Ma VAFFANCULO!!!
          if (nrow(X_OOS) > 0){
            
            # --- Kernel-Logistic-Regression
            K_test <- predict_klr(out_KLR,train_embeddings, test_embeddings)$K
            K_OOS <- predict_klr(out_KLR,train_embeddings, oos_embeddings)$K
            NNkbc_klr_PROBA_contrib_shit[,,j] <- rbind(t(KLR_contributions(out_KLR$alpha,K_test,out_KLR$intercept)$proba),
                                                       t(KLR_contributions(out_KLR$alpha,K_OOS,out_KLR$intercept)$proba))
            NNkbc_klr_LOGODDS_contrib_shit[,,j] <- rbind(t(KLR_contributions(out_KLR$alpha,K_test,out_KLR$intercept)$log_odds),
                                                         t(KLR_contributions(out_KLR$alpha,K_OOS,out_KLR$intercept)$log_odds))
            
            
          } else {
            # --- Kernel-Logistic-Regression
            K_test <- predict_klr(out_KLR,train_embeddings, test_embeddings)$K
            NNkbc_klr_PROBA_contrib_shit[,,j] <- t(KLR_contributions(out_KLR$alpha,K_test,out_KLR$intercept)$proba)
            NNkbc_klr_LOGODDS_contrib_shit[,,j] <- t(KLR_contributions(out_KLR$alpha,K_test,out_KLR$intercept)$log_odds)
            
          }
          
          
        }
        
        
        
        
      } else if (nn_hyps$objective == 'regression'){
        
        # --- Optimize Lambda for Kernel-Estimation
        mse_lambda <- c()
        for (la in nn_hyps[['lambda_grid']]){
          
          # --- Compute the regularization term
          regularization_term_la <- diag(nrow(x_train)) * la
          K_in_la =  solve(regularization_term_la + train_crossprod)
          
          # --- In-Sample-Prediction
          yhat_ins_kernel <- invert_scaling((K_in_la %*% K_in_la) %*% Y, temp)
          mse_lambda <- c(mse_lambda, mean((pred.in.ensemble[,j] - yhat_ins_kernel)^2))
          
        }
        
        # --- Which was the optimal lambda?
        mse_opt[j] <- min(mse_lambda)
        lambda_opt[j] <- nn_hyps[['lambda_grid']][which(mse_lambda == min(mse_lambda))]
        
        
        
        # Compute the regularized inverse of the training cross-product
        regularization_term <- diag(nrow(x_train)) * lambda_opt[j]
        K.in =  solve(regularization_term + train_crossprod)
        
        # Compute the final result
        result <- K.out %*% K.in
        inner.prod[,,j] <- result 
        
        # --- Store the DUAL Solution
        pred.k[,j] = invert_scaling(as.matrix(inner.prod[,,j]%*%Y),temp)
      }
      
    } else {
      
      pred.in.ensemble[,j] <- as.matrix(model(x_train[,])[[1]])
      pred.ensemble[,j] <- as.matrix(model(x_test)[[1]])
      if (nrow(X_OOS) > 0){
        pred.OOS.ensemble[,j] <- as.matrix(model(x_OOS)[[1]])
      }
      
      
      
      
      # Convert the model outputs to matrices
      train_embeddings <- as.matrix(model(x_train)[[2]])
      test_embeddings <- as.matrix(model(x_test)[[2]])
      if (nrow(X_OOS) > 0){
        oos_embeddings <- as.matrix(model(x_OOS)[[2]])
      }
      
      # Compute the cross-product of the training embeddings
      train_crossprod <- tcrossprod(train_embeddings)
      
      #print(dim(train_embeddings))
      #print(dim(test_embeddings))
      
      # Compute the kernel matrix
      K.out <- tcrossprod(test_embeddings,train_embeddings)
      if (nrow(X_OOS) > 0){
        K.oos <- tcrossprod(oos_embeddings,train_embeddings)
      }
      
       
      
      
      # --- Find the DUAL Solution:
      if (nn_hyps$objective == 'binary_classification'){
      
        # -------------------------------- Prediction via Kernel-Logistic-Regression ------------------------------- #
        
        # --- Run the Grid-Search SHIT! This is pure pain!
        mse_lambda <- mse_oos_lambda <- c()
        conv_list_j <- log_loss_list_j <- list()
        count <- 0
        for (la in nn_hyps[['lambda_grid_KLR']]){
          count <- count+1
          # --- Fitting:
          out_KLR <- kernel_logistic_regression(train_embeddings,Y,
                                                kernel_params=list(type='linear'),
                                                lambda = la, 
                                                optimizer = nn_hyps[['optim_KLR']],
                                                maxit = nn_hyps[['maxit_KLR']], lr = nn_hyps[['lr_KLR']], eps=1e-15,
                                                verbose=FALSE)
          
          conv_list_j[[count]] <- out_KLR$fi
          log_loss_list_j[[count]] <- out_KLR$logloss
          
          # --- Prediction: In-Sample
          yhat_ins_klr <- predict_klr(out_KLR,train_embeddings, train_embeddings)$predictions
          
          mse_lambda <- c(mse_lambda, mean((pred.in.ensemble[,j] - yhat_ins_klr)^2)/var(pred.in.ensemble[,j]))
          
          # --- Prediction: Out-Of-Sample
          yhat_oos_klr <- predict_klr(out_KLR,train_embeddings, test_embeddings)$predictions
          
          mse_oos_lambda <- c(mse_oos_lambda, mean((pred.ensemble[,j] - yhat_oos_klr)^2)/var(pred.ensemble[,j]))
          
          
        }
        
        # --- Which was the optimal lambda?
        mse_opt_KLR[j] <- min(mse_lambda)
        lambda_opt_KLR[j] <- nn_hyps[['lambda_grid_KLR']][which(mse_lambda == min(mse_lambda))][1]
        
        mse_ins_KLR[,j] <- mse_lambda
        mse_oos_KLR[,j] <- mse_oos_lambda
        
        # --- Fitting, using the 'optimal' lambda
        out_KLR <- kernel_logistic_regression(train_embeddings,Y,
                                              kernel_params=list(type='linear'),
                                              lambda = lambda_opt_KLR[j], 
                                              optimizer = nn_hyps[['optim_KLR']],
                                              maxit = nn_hyps[['maxit_KLR']], lr = nn_hyps[['lr_KLR']], eps=1e-15,
                                              verbose=TRUE)
        
        
        
        if (out_KLR$algo_conv | nn_hyps[['keep_all']]) { #out_KLR$algo_conv
          
          conv_list_j[[count+1]] <- out_KLR$fi
          log_loss_list_j[[count+1]] <- out_KLR$logloss
          
          names(conv_list_j) <- names(log_loss_list_j) <- c(paste0('la',nn_hyps[['lambda_grid_KLR']]),'final')
          conv_list[[j]] <- conv_list_j
          log_loss_list[[j]] <- log_loss_list_j
          
          par(mfrow=c(3,2))
          ts.plot(na.omit(out_KLR$fi[-c(1:10)]), main = paste0("convergence, la",lambda_opt_KLR[j]), ylab="",xlab="")
          ts.plot(na.omit(log(out_KLR$fi)), main="log convergence", ylab="",xlab="")
          ts.plot(na.omit(out_KLR$logloss[-c(1:10)]), main="logloss", ylab="",xlab="")
          ts.plot(na.omit(log(out_KLR$logloss[-c(1:10)])), main="loglogloss", ylab="",xlab="")
          #ts.plot(mse_lambda, main="MSE INS", ylab = "", xlab="lambda")
          #ts.plot(mse_oos_lambda, main="MSE OOS", ylab = "", xlab="lambda")
          plot(x = as.factor(nn_hyps[['lambda_grid_KLR']]), y = mse_lambda, main="MSE INS", ylab = "", xlab = "lambda")
          plot(x = as.factor(nn_hyps[['lambda_grid_KLR']]), y = mse_oos_lambda, main="MSE OOS", ylab = "", xlab = "lambda")
          
          # --------------------- Collect the results ----------------------- #
          
          
          # --- Training Set
          pred.in.k_klr[,j] = predict_klr(out_KLR,train_embeddings, train_embeddings)$predictions
          
          # --- Test Set
          pred.k_klr[,j] = predict_klr(out_KLR,train_embeddings, test_embeddings)$predictions
          
          # --- OOS
          if (nrow(X_OOS) > 0){
            pred.OOS.k_klr[,j] = predict_klr(out_KLR,train_embeddings, oos_embeddings)$predictions
          }
          
          
          # --- Non solo le maledette previste ma anche i contribuzioni! Ma VAFFANCULO!!!
          if (nrow(X_OOS) > 0){
            
            # --- Kernel-Logistic-Regression
            K_test <- predict_klr(out_KLR,train_embeddings, test_embeddings)$K
            K_OOS <- predict_klr(out_KLR,train_embeddings, oos_embeddings)$K
            NNkbc_klr_PROBA_contrib_shit[,,j] <- rbind(t(KLR_contributions(out_KLR$alpha,K_test,out_KLR$intercept)$proba),
                                                       t(KLR_contributions(out_KLR$alpha,K_OOS,out_KLR$intercept)$proba))
            NNkbc_klr_LOGODDS_contrib_shit[,,j] <- rbind(t(KLR_contributions(out_KLR$alpha,K_test,out_KLR$intercept)$log_odds),
                                                         t(KLR_contributions(out_KLR$alpha,K_OOS,out_KLR$intercept)$log_odds))
            
            
          } else {
            # --- Kernel-Logistic-Regression
            K_test <- predict_klr(out_KLR,train_embeddings, test_embeddings)$K
            NNkbc_klr_PROBA_contrib_shit[,,j] <- t(KLR_contributions(out_KLR$alpha,K_test,out_KLR$intercept)$proba)
            NNkbc_klr_LOGODDS_contrib_shit[,,j] <- t(KLR_contributions(out_KLR$alpha,K_test,out_KLR$intercept)$log_odds)
            
          }
          
          
        }
      
      } else if (nn_hyps$objective == 'regression'){
        
        
        
        # --- Optimize Lambda for Kernel-Estimation
        mse_lambda <- c()
        for (la in nn_hyps[['lambda_grid']]){
          
          # --- Compute the regularization term and make OOS-prediction
          regularization_term_la <- diag(nrow(x_train)) * la
          K_in_la =  solve(regularization_term_la + train_crossprod)
          
          # --- In-Sample-Prediction
          yhat_ins_kernel <- (K_in_la %*% K_in_la) %*% Y
          mse_lambda <- c(mse_lambda, mean((pred.in.ensemble[,j] - yhat_ins_kernel)^2))
          
          
        }
        
        # --- Which was the optimal lambda?
        mse_opt[j] <- min(mse_lambda)
        lambda_opt[j] <- nn_hyps[['lambda_grid']][which(mse_lambda == min(mse_lambda))]
        
        
        # Compute the regularized inverse of the training cross-product --- based on the optimal lambda
        regularization_term <- diag(nrow(x_train)) * lambda_opt[j]
        K.in =  solve(regularization_term + train_crossprod)
        
        # Compute the final result
        result <- K.out %*% K.in
        inner.prod[,,j] <- result
        
        
        # --- Store the DUAL Solution
        pred.k[,j] = as.matrix(inner.prod[,,j]%*%Y)
      }
      
      
    }
    
    if(show_train==2) {
      setTxtProgressBar(pb, j)
    }
    
  } # j, bootstrap end
  
  if(show_train==2) {
    close(pb)
  }
  
  # Output
  pred.in <- rowMeans(pred.in.ensemble, na.rm = T)
  pred <- rowMeans(pred.ensemble, na.rm = T)
  
  
  if (nn_hyps$objective == 'binary_classification'){
    
    
    results <- list(pred.in.ensemble=pred.in.ensemble,
                    pred.ensemble=pred.ensemble,
                    pred.OOS.ensemble=pred.OOS.ensemble,
                    pred.in=pred.in,
                    pred=pred,
                    pred.OOS = rowMeans(pred.OOS.ensemble,na.rm = T),
                    pred.in.kernel_klr.ensemble=pred.in.k_klr,
                    pred.in.kernel_klr=rowMeans(pred.in.k_klr,na.rm = T),
                    pred.kernel_klr.ensemble = pred.k_klr,
                    pred.kernel_klr = rowMeans(pred.k_klr,na.rm = T),
                    pred.OOS.kernel_klr = rowMeans(pred.OOS.k_klr,na.rm = T),
                    pred.OOS.kernel_klr.ensemble = pred.OOS.k_klr,
                    NNkbc_klr_LOGODDS_contrib = apply(NNkbc_klr_LOGODDS_contrib_shit,c(1,2),mean,na.rm = T),
                    NNkbc_klr_PROBA_contrib = apply(NNkbc_klr_PROBA_contrib_shit,c(1,2),mean,na.rm = T),
                    NNkbc_klr_LOGODDS_contrib_shit_all = NNkbc_klr_LOGODDS_contrib_shit,
                    NNkbc_klr_PROBA_contrib_shit_all = NNkbc_klr_PROBA_contrib_shit,
                    lambda_opt_SIGMOID=lambda_opt,
                    mse_opt_SIGMOID=mse_opt,
                    lambda_opt_KLR=lambda_opt_KLR,
                    mse_opt_KLR=mse_opt_KLR,
                    trained_model=trained_model,
                    scaler = temp,
                    standardize=standardize,
                    conv_list = conv_list,
                    log_loss_list = log_loss_list,
                    mse_ins_KLR = mse_ins_KLR,
                    mse_oos_KLR = mse_oos_KLR,
                    intercept = out_KLR$intercept
    )
    
  } else if (nn_hyps$objective == 'regression'){
    
    results <- list(pred.in.ensemble=pred.in.ensemble,
                    pred.ensemble=pred.ensemble,
                    pred.in=pred.in,
                    portfolio.weights = apply(inner.prod,c(1,2),mean,na.rm = T), 
                    pred.kernel = rowMeans(pred.k),
                    pred=pred,
                    lambda_opt=lambda_opt,
                    mse_opt=mse_opt,
                    trained_model=trained_model,
                    scaler = temp,
                    standardize=standardize)
  }
  
  return(results)
  
} # MLP FUNCTION END

# =====================================================================================================
## PREDICT
# =====================================================================================================

predict_nn <- function(mlp,Xtest,nn_hyps) {
  
  
  EmptyNN <- function(nn_hyps) {
    # Build Model
    net = nn_module(
      "nnet",
      
      initialize = function(n_features=nn_hyps$n_features, nodes=nn_hyps$nodes,dropout_rate=nn_hyps$dropout_rate){
        
        self$n_layers = length(nodes)
        
        self$input = nn_linear(n_features,nodes[1])
        
        if(length(nodes)==1) {
          self$first = nn_linear(nodes,nodes)
        }else{
          self$first = nn_linear(nodes[1],nodes[1])
          self$hidden <- nn_module_list(lapply(1:(length(nodes)-1), function(x) nn_linear(nodes[x], nodes[x+1])))
        }
        
        self$output = nn_linear(nodes[length(nodes)],1)
        self$dropout = nn_dropout(p=dropout_rate)
        
      },
      
      forward = function(x){
        
        # Input
        x = torch_relu(self$input(x))
        
        # Hidden
        x = torch_relu(self$first(x))
        x = self$dropout(x)
        
        if(self$n_layers>1) {
          
          for(layer in 1:(self$n_layers-1)) {
            x = torch_relu(self$hidden[[layer]](x))
            x = self$dropout(x)
          }
          
        }
        
        # Output
        yhat = torch_squeeze(self$output(x))
        
        result <- list(yhat)
        return(result)
      }
    )
    
    model = net()
    return(model)
  }
  
  
  # Create empty matrix
  num_bootstrap <- dim(mlp$pred.in.ensemble)[2]
  
  if(!is.null(nrow(Xtest))) {
    obs <- nrow(Xtest)
  } else{
    obs <- 1
  }
  
  forecasts <- matrix(data = NA, nrow = obs, ncol = num_bootstrap)
  forecasts[,] <- NA
  
  if(mlp$standardize == T) {
    scaler <- mlp$scaler
    newx <- predict_scale_data(scaler, Xtest)
  } else {
    newx <- Xtest
  }
  
  
  for(i in 1:num_bootstrap) {
    
    # Format data
    Xtest <- torch_tensor(newx, dtype = torch_float(), requires_grad = F)
    
    # Load trained models
    state_dict <- mlp$trained_model[[i]]
    state_model <- lapply(state_dict$state_dict(), function(x) x$clone()) 
    model <- EmptyNN(nn_hyps)
    model$load_state_dict(state_model)
    
    # Predict
    model$eval()
    if(mlp$standardize == T) {
      
      forecasts[,i] <- invert_scaling(as.matrix(model(Xtest)[[1]]),scaler)
      
    }else{
      
      forecasts[,i] <- as.matrix(model(Xtest)[[1]])
      
    }
    
  }
  
  return(forecasts)
  
}

# =====================================================================================================
## STANDARDIZATION
# =====================================================================================================

scale_data <- function(Xtrain, Ytrain, Xtest, Ytest) {
  
  # Features
  sigma_x <- apply(Xtrain,2,sd)
  mu_x <- apply(Xtrain,2,mean)
  
  if(is.null(dim(Xtest))==TRUE) {
    dim(Xtest) <- c(1, length(Xtest))
  }
  
  Xtest <- do.call(cbind,lapply(1:length(mu_x),function(x) (Xtest[,x] - mu_x[x])/sigma_x[x]))
  Xtrain <- do.call(cbind,lapply(1:length(mu_x),function(x) (Xtrain[,x] - mu_x[x])/sigma_x[x]))
  
  # Target
  sigma_y <- sd(Ytrain)
  mu_y <- mean(Ytrain)
  
  Ytrain <- (Ytrain-mu_y)/sigma_y
  Ytest <- (Ytest-mu_y)/sigma_y
  
  return(list(Xtrain = Xtrain, Ytrain = Ytrain, Xtest = Xtest, Ytest = Ytest , sigma_y = sigma_y, mu_y = mu_y,
              sigma_x = sigma_x, mu_x = mu_x))
  
}

invert_scaling <- function(scaled, scaler) {
  
  sigma_y <- scaler$sigma_y
  mu_y <- scaler$mu_y
  
  
  inverted <- sigma_y*scaled + mu_y
  
  return(inverted)
  
}

rescale_data <- function(results,newx) {
  
  # Features
  sigma_x <- results$scaler$sigma_x
  mu_x <- results$scaler$mu_x
  
  if(is.null(dim(newx))==TRUE) {
    dim(newx) <- c(1, length(newx))
  }
  
  Xtest <- do.call(cbind,lapply(1:length(mu_x),function(x) (newx[,x] - mu_x[x])/sigma_x[x]))
  
  return(newx = Xtest)
  
}

predict_scale_data <- function(scaler, Xtest) {
  
  # Features
  sigma_x <- scaler$sigma_x
  mu_x <- scaler$mu_x
  
  if(is.null(dim(Xtest))==TRUE) {
    dim(Xtest) <- c(1, length(Xtest))
  }
  
  Xtest <- do.call(cbind,lapply(1:length(mu_x),function(x) (Xtest[,x] - mu_x[x])/sigma_x[x]))
  
  return(Xtest)
}

sigmoid <- function(x){
  return(1/(1+exp(-x)))
}

bce_loss <- function(y_true,y_hat){
  return(-sum(y_true*log(y_hat) + (1-y_true)*log(1-y_hat)))
}
