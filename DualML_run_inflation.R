#################################################################################################################
#
#
#                       Goulet Coulombe, Philippe, and Goebel, Maximilian, and Klieber, Karin
#
#                             "Dual Interpretation of Machine Learning Forecasts"
#
#                            APPLICATION: Post-Pandemic Inflation (h=1 step ahead)
#
#
#################################################################################################################

# --- This version: 2024-11-17


# ---------------------------------------   Load packages ---------------------------------------- #
library(ranger)
library(lightgbm)
library(pracma)
library(kernlab)
library(glmnet)

directory <- '~/Dropbox/DualRoute/02_code/50_Max/002_packaging/'
source(file.path(directory,'/DualML.R'))
source(file.path(directory,'/MLP.R'))


# ----------------------------------------   Auxiliary Functions ---------------------------------------- #
block.sampler <- function(X,sampling_rate,block_size,num.tree,idx_pos=c()){
  
  inbag=list()
  inbag2=list()
  
  for(j in 1:num.tree){
    sample_index <- c(1:nrow(X))
    groups<-sort(base::sample(x=c(1:(length(sample_index)/block_size)),size=length(sample_index),replace=TRUE))
    rando.vec <- rexp(rate=1,n=length(sample_index)/block_size)[groups] +0.1
    
    # --- Potential stratification:
    if (length(idx_pos) > 0){
      
      pos_sample_true <- groups[idx_pos]
      
      # --- Sample positive-groups
      pos_train <- c()
      pos_sample <- unique(pos_sample_true)
      while (length(pos_train) < ((length(pos_sample_true) * sampling_rate) - block_size/2) ){
        
        take_pos <- sample(pos_sample,size=1)
        pos_train <- c(pos_train, rep(take_pos,times=length(which(pos_sample_true == take_pos))))
        pos_sample <- pos_sample[-which(pos_sample == take_pos)]
        
      }
      
      pos_val <- pos_sample_true[-which(pos_sample_true %in% unique(pos_train))]
      
      # --- Up-weight groups in 'pos_train'
      rando.vec[which(groups %in% unique(pos_train))] <- rando.vec[which(groups %in% unique(pos_train))] + abs(max(rando.vec))
      # --- Down-weight groups NOT 'pos_train'
      rando.vec[which(groups %in% unique(pos_val))] <- rando.vec[which(groups %in% unique(pos_val))] - abs(max(rando.vec))
    }
    
    
    chosen.ones.plus<-rando.vec
    rando.vec<-which(chosen.ones.plus>quantile(chosen.ones.plus,1-sampling_rate))
    chosen.ones.plus<-sample_index[rando.vec]
    
    boot <- c(sort(chosen.ones.plus))                         # training
    oob <- c(1:nrow(X))[-boot]   
    count = 1:nrow(X)
    inbagj = I(is.element(count,boot))
    inbag[[j]] = as.numeric(inbagj)
  }
  
  for(j in 1:num.tree){
    sample_index <- c(1:nrow(X))
    groups<-sort(base::sample(x=c(1:(length(sample_index)/block_size)),size=length(sample_index),replace=TRUE))
    rando.vec <- rexp(rate=1,n=length(sample_index)/block_size)[groups] +0.1
    chosen.ones.plus<-rando.vec
    rando.vec<-which(chosen.ones.plus>quantile(chosen.ones.plus,1-sampling_rate))
    chosen.ones.plus<-sample_index[rando.vec]
    
    boot <- c(sort(chosen.ones.plus))                         # training
    oob <- c(1:nrow(X))[-boot]   
    count = 1:nrow(X)
    inbagj = I(is.element(count,boot))
    inbag2[[j]] = as.numeric(inbagj)
  }
  
  return(list(inbag1=inbag,inbag2=inbag2))
}


# ----------------------------------------   Prepare Data ---------------------------------------- #
# --- Load data
data <- read.csv(paste0(directory,'US_data.csv'))
data$Date <- as.Date(data$Date, format='%m/%d/%Y')

# --- Remove some stuff
remove_var <- c( "outputGap", "chan_outputgap")
data <- data[, !apply(sapply(remove_var, function(x) grepl(x, colnames(data))), 1, any)]

# --- Define in- and out-of-sample sets
idx_ins <- which(data$Date <= '2019-12-31')
idx_oos <- which(data$Date > '2019-12-31')

# --- Get the scalers:
scaler_mean <- apply(data[idx_ins,3:ncol(data)],2,mean)
scaler_sd <- apply(data[idx_ins,3:ncol(data)],2,sd)

# --- Define 'Ytrain':
Ytrain <- data$y[idx_ins]

# --- Define 'Xtrain':
Xtrain <- data.frame(sapply(1:length(scaler_mean), function(x) (data[idx_ins, x+2] - scaler_mean[x]) / scaler_sd[x]))

# --- Define 'Ytest':
Ytest <- data$y[idx_oos]

# --- Define 'Xtest':
Xtest <- data.frame(sapply(1:length(scaler_mean), function(x) (data[idx_oos, x+2] - scaler_mean[x]) / scaler_sd[x]))



# ----------------------------------------   00. OLS - e.g., FAAR4 ---------------------------------------- #
# --- 00.1  Get the column-indices for the 'L0_' variables
pca_index <- grep('L0_',colnames(data))
# --- 00.2  Now exclude the first 4 indices as they indicate 'L0_F_US[1-4]' 
pca_index <- pca_index[5:length(pca_index)]
# --- 00.3  For Training: run PCA on the training-data ONLY
pca_train <- prcomp(x=data[idx_ins,pca_index],center=T,scale=T, rank.=4)
pc_train <- pca_train$x
# --- 00.4  For Testing: run PCA on the training- PLUS test-data
pc_test <- setNames(data.frame(matrix(NA,nrow=0,ncol=4)),nm=paste0('PC',c(1:4)))
for (ll in 1:length(idx_oos)){
  # --- Run PCA for 'll'th OOS period:
  pca_test <- prcomp(x=data[c(idx_ins,idx_oos[1:ll]),pca_index],center=T,scale=T, rank.=4)
  # --- Collect the components
  pc_test <- setNames(rbind(pc_test,pca_test$x[idx_oos[ll],]),nm=paste0('PC',c(1:4)))
}
# --- 00.5  Stack the training- & test-PCs
pc_train_test <- setNames(rbind(pc_train,pc_test),nm=paste0('L0_PC',c(1:4)))
pc_train_test$Date <- data$Date[c(idx_ins,idx_oos)]
# --- 00.6 Lag them
pc_train_test[,paste0('L1_PC',c(1:4))] <- rbind(matrix(NA,nrow=1,ncol=4,dimnames=list(c(),paste0('L0_PC',c(1:4)))),
                                                pc_train_test[1:(nrow(pc_train_test)-1),paste0('L0_PC',c(1:4))])
# --- 00.7 Add lags of the target and merge
data_pca <- merge(data[,c(1,which(colnames(data) == 'y'),grep('L_',colnames(data)))],pc_train_test, by='Date',all.x=T)
data_pca <- na.omit(data_pca)
# --- 00.8 Get Xtrain and Xtest
X_ar_train <- as.matrix(data_pca[which(data_pca$Date <= '2019-12-31'),-c(1:2)])
y_ar_train <- data_pca[which(data_pca$Date <= '2019-12-31'),'y']
X_ar_test <- as.matrix(data_pca[which(data_pca$Date > '2019-12-31'),-c(1:2)])
X_ar_train <- cbind('intercept'=rep(1,times=nrow(X_ar_train)),X_ar_train)
X_ar_test <- cbind('intercept'=rep(1,times=nrow(X_ar_test)),X_ar_test)

# --- 00.9 Get Dual interpretation
backpack <- list('type' = 'OLS',
                 'params' = list('Xtrain'=X_ar_train[7:nrow(X_ar_train),],
                                 'Xtest'=X_ar_test,
                                 'Ytrain'=y_ar_train[7:nrow(X_ar_train)],
                                 'dates_ins'=data[idx_ins[7:nrow(X_ar_train)],'Date'],
                                 'model_object'=NULL,
                                 'intercept'=T)
)
test_FAAR <- DualML(run__model=backpack)

# --- 00.10 Plot: weights
plot(x=test_FAAR[['weights']]['date'], y=test_FAAR[['weights']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('FAAR: Observation Weights -- OOS-Period: ',1),col=2)

# --- 00.11 Plot: contributions
plot(x=test_FAAR[['weights']]['date'], y=test_FAAR[['contributions']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('FAAR: Observation Contributions -- OOS-Period: ',1),col=4)


# -------------------------------------------   1. Random Forest ---------------------------------------- #
# --- 1.0 Hyperparameters
my_blocksize <- 8
N_trees <- 500
mtry_denom <- 3
my_minnodesize <- 5
my_samplefraction <- 0.75

# --- 1.1 Draw blocks of observations for Out-Of-Bag computations
bs <- block.sampler(Xtrain,  sampling_rate=0.8, block_size = my_blocksize, num.tree=N_trees)

# --- 1.2 Grow the trees
mod_rf <- ranger(Y ~ ., data = cbind(Y=Ytrain,Xtrain),
                 num.trees=N_trees,
                 inbag=bs$inbag1, 
                 mtry = round(ncol(Xtrain)/mtry_denom),
                 min.node.size = my_minnodesize, 
                 sample.fraction = my_samplefraction,
                 keep.inbag=TRUE, # --- This is optional! ---> no it's not anymore!!! (v240621)
                 num.threads = 1)


# --- 1.3 Get Dual Interpretation
backpack <- list('type' = 'RF',
                 'params' = list('Xtrain'=Xtrain,
                                 'Xtest'=Xtest,
                                 'Ytrain'=Ytrain,
                                 'dates_ins'=data[idx_ins,'Date'],
                                 'model_object'=mod_rf)
)
test_RF <- DualML(backpack)


# --- 1.4 Plot: weights
plot(x=test_RF[['weights']]['date'], y=test_RF[['weights']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('RF: Observation Weights -- OOS-Period: ',1),col=2)

# --- 1.5 Plot: contributions
plot(x=test_RF[['weights']]['date'], y=test_RF[['contributions']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('RF: Observation Contributions -- OOS-Period: ',1),col=4)


# ----------------------------------------   2. Boosted Trees (LightGBM) ---------------------------------------- #
# --- 2.0 Set the Hyperparameters
params_lgb <- list("boosting_type"="gbdt",
                   "objective"= "regression",
                   "metric"= "rmse",
                   "num_leaves"= 14,
                   "verbose"= 0,
                   "min_data"= 2,
                   "learning_rate"= 0.01)

# --- 2.1 Use a block-bootstrapped validation set
set.seed(123)
bs_lgb <- block.sampler(Xtrain,  sampling_rate=1, block_size = my_blocksize, num.tree=1)

# --- 2.2 Run LightGBM
Dtrain_lgb <- lgb.Dataset(data=as.matrix(Xtrain[bs_lgb$inbag1[[1]] == 1,]), label=Ytrain[bs_lgb$inbag1[[1]] == 1])
Dval_lgb <- lgb.Dataset(data=as.matrix(Xtrain[bs_lgb$inbag1[[1]] == 0,]), label=Ytrain[bs_lgb$inbag1[[1]] == 0])

mod_lgb <- lgb.train(data=Dtrain_lgb,
                     params=params_lgb,
                     nrounds=100, # --- number of trees, default=100
                     #early_stopping_round = 20,
                     valid=list(test=Dval_lgb),
                     verbose=FALSE)

# --- 2.3 Get Dual Interpretation
backpack <- list('type' = 'LGB',
                 'params' = list('Xtrain'=Xtrain,
                                 'Xtest'=Xtest,
                                 'Ytrain'=Ytrain,
                                 'dates_ins'=data[idx_ins,'Date'],
                                 'model_object'=mod_lgb)
)
test_LGB <- DualML(backpack)

# --- 2.4 Plot: weights
plot(x=test_LGB[['weights']]['date'], y=test_LGB[['weights']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('LGB: Observation Weights -- OOS-Period: ',1),col=2)

# --- 2.5 Plot: contributions
plot(x=test_LGB[['weights']]['date'], y=test_LGB[['contributions']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('LGB: Observation Contributions -- OOS-Period: ',1),col=4)


# ----------------------------------------   3. Kernel Ridge-Regression ---------------------------------------- #
# --- 3.0 Hyperparameters
krr_kernel <- 'laplace'
krr_kernel_sigma <- 1e-04
krr_lmbda <- 1e-05


# --- 3.1 Set the kernel & Estimate
if (krr_kernel == 'gaussian'){
  K_xx <- rbfdot(sigma=krr_kernel_sigma) 
} else if (krr_kernel == 'laplace'){
  K_xx <- laplacedot(sigma=krr_kernel_sigma)
}

# --- 3.2 Calculate the Kernel-Matrix
K_train <- kernelMatrix(K_xx, as.matrix(Xtrain))
K_test <- kernelMatrix(K_xx, as.matrix(Xtest), as.matrix(Xtrain))

# --- 3.3 Get Dual Interpretation
backpack <- list('type' = 'KRR',
                 'params' = list('Xtrain'=K_train,
                                 'Xtest'=K_test,
                                 'Ytrain'=Ytrain,
                                 'dates_ins'=data[idx_ins,'Date'],
                                 'model_object'=NULL,
                                 'lmbda'=krr_lmbda)
)
test_KRR <- DualML(backpack)

# --- 3.4 Plot: weights
plot(x=test_KRR[['weights']]['date'], y=test_KRR[['weights']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('KRR: Observation Weights -- OOS-Period: ',1),col=2)

# --- 3.5 Plot: contributions
plot(x=test_KRR[['weights']]['date'], y=test_KRR[['contributions']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('KRR: Observation Contributions -- OOS-Period: ',1),col=4)



# ----------------------------------------   4. Ridge Regression ---------------------------------------- #
# --- 4.0 Scale the target
Ytrain_rr <- Ytrain
Ytest_rr <- Ytest

y_mu_rr <- mean(Ytrain)
y_sd_rr <- sd(Ytrain)

Ytrain_rr <- (Ytrain_rr - y_mu_rr) / y_sd_rr

# --- 4.1 Cross-Validate Lambda
# --- --- Create the K-folds:
cv_folds <- as.vector(sapply(c(1:9), function(x) rep(x,length(Ytrain_rr)/10)))
cv_folds <- c(cv_folds, rep(10,length(Ytrain_rr)-length(cv_folds)))

lmbda_grid <- seq(1e-3,100,length.out=1000)
intercept_RR <- FALSE
mod_RR_cv <- cv.glmnet(as.matrix(Xtrain),
                       Ytrain_rr,
                       intercept=intercept_RR,
                       alpha=0,
                       nfolds=10,
                       type.measure='mse',
                       standardize=F,standardize.response=F,
                       foldid=cv_folds,
                       lambda = lmbda_grid)

# --- 4.2 Extract the optimal Lambda:
lmbda_RR <- mod_RR_cv$lambda.1se

# --- 4.3 Get Dual Interpretation
backpack <- list('type' = 'RR',
                 'params' = list('Xtrain'=Xtrain,
                                 'Xtest'=Xtest,
                                 'Ytrain'=Ytrain,
                                 'dates_ins'=data[idx_ins,'Date'],
                                 'model_object'=NULL,
                                 'lmbda'=lmbda_RR,
                                 'intercept'=intercept_RR)
)
test_RR <- DualML(backpack)

# --- 4.4 Plot: weights
plot(x=test_RR[['weights']]['date'], y=test_RR[['weights']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('RR: Observation Weights -- OOS-Period: ',1),col=2)

# --- 4.5 Plot: contributions
plot(x=test_RR[['weights']]['date'], y=test_RR[['contributions']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('RR: Observation Contributions -- OOS-Period: ',1),col=4)



# ----------------------------------------   5. NN ---------------------------------------- #

# --- 5.1 Set the Hyperparameters
nn_hyps <- list(n_features=ncol(Xtrain),
                nodes=rep(400,3),      # same number of nodes in every layers
                patience=10,           # Return the best model
                epochs=100,
                lr=0.001,
                tol=0.01,
                show_train=2,          # 1=show each bootstrap loss, 2=progress bar, 3+=show nothing
                num_average=30,        # Number of bootstraps
                dropout_rate=0.2,
                sampling_rate = 0.85,
                batch_size = 32,
                num_batches = NA,
                lambda_grid = seq(1e-3,50,length.out=200) # --- Set the grid for searching for the optimal lambda for the kernel-prediction)
)

# --- 5.2 Run the NN
mod_NN <- MLP(X = as.matrix(Xtrain), 
              Y = c(Ytrain),
              Xtest = as.matrix(Xtest), 
              Ytest = c(Ytest),
              X_OOS = as.matrix(Xtest),
              nn_hyps = nn_hyps,
              standardize=T,
              seed=1234)
  
# --- 5.3 Get Dual Interpretation
backpack <- list('type' = 'NN',
                 'params' = list('Xtrain'=NULL,
                                 'Xtest'=NULL,
                                 'Ytrain'=Ytrain,
                                 'dates_ins'=data[idx_ins,'Date'],
                                 'model_object'=mod_NN,
                                 'lmbda'=NULL,
                                 'intercept'=NULL)
)
test_NN <- DualML(backpack)

# --- 5.4 Plot: weights
plot(x=test_NN[['weights']]['date'], y=test_NN[['weights']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('RR: Observation Weights -- OOS-Period: ',1),col=2)

# --- 5.5 Plot: contributions
plot(x=test_NN[['weights']]['date'], y=test_NN[['contributions']][,1], 
     type='l',xlab='Date',ylab='',main=paste0('NN: Observation Contributions -- OOS-Period: ',1),col=4)



# ----------------------------------------   Evaluating Results ---------------------------------------- #
# --- Evaluation: concentration
test_LGB$concentration[c(3,5,10)]
test_RF$concentration[c(3,5,10)]
test_NN$concentration[c(3,5,10)]
test_KRR$concentration[c(3,5,10)]
test_RR$concentration[c(3,5,10)]
test_FAAR$concentration[c(3,5,10)]

# --- Evaluation: leverage
test_LGB$leverage[c(3,5,10)]
test_RF$leverage[c(3,5,10)]
test_NN$leverage[c(3,5,10)]
test_KRR$leverage[c(3,5,10)]
test_RR$leverage[c(3,5,10)]
test_FAAR$leverage[c(3,5,10)]


# --- Evaluation: short_position
test_LGB$short_position[c(3,5,10)]
test_RF$short_position[c(3,5,10)]
test_NN$short_position[c(3,5,10)]
test_KRR$short_position[c(3,5,10)]
test_RR$short_position[c(3,5,10)]
test_FAAR$short_position[c(3,5,10)]

# --- Evaluation: turnover
test_LGB$turnover
test_RF$turnover
test_NN$turnover
test_KRR$turnover
test_RR$turnover
test_FAAR$turnover



