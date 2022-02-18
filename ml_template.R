library(tidyverse)
library(caret)
library(randomForest)


# specify params

d <- ##data
train_perc <- ##train%
seed_num <- ##
Y <- ##label

# split data

set.seed(seed_num)
trn_idx = d[[Y]] %>% createDataPartition(p = train_perc, list = FALSE)
dtrn = d[trn_idx,]
dtst = d[-trn_idx,]

# estimate preprocessing parameters

preproc_par <- train.data %>% preProcess(method = c("center", "scale"))

# transform data using the estimated parameters

dtrn.proc <- preproc_par %>% predict(dtrn)
dtst.proc <- preproc_par %>% predict(dtst)
d.proc <- preproc_par %>% predict(d)


## random forest

tunegrid <- expand.grid(.mtry=c(1:20))
control <- trainControl(method = "cv",
                        number = 5,
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary,
                        savePredictions = "final")
                        
mod_rf <- train(!!sym(Y) ~ .,
                data = d,
                method = 'rf',
                metric = 'Accuracy',
                type = "Classification",
                tuneGrid = tunegrid,
                trlControl = control)


mtry_bst <- unlist(mod_rf$bestTune)

mod_rf <- randomForest(!!sym(Y) ~ ., data = dtrn, importance = TRUE, mtry=mtry_bst)

# Confusion Matrix of the train/test set:

trn_res <- confusionMatrix(predict(mod_rf, dtrn[-1]), dtrn[[Y]]) 
trn_res$byClass[,c("Sensitivity", "Specificity", "Precision")]

tst_res <- confusionMatrix(predict(mod_rf, dtst[-1]), dtst[[Y]]) 
tst_res$byClass[,c("Sensitivity", "Specificity", "Precision")]

## kNN

num_group <- 4
mod_knn <- knn(dtrn.proc[,-1], dtst.proc[,-1], dtrn.proc[[Y]], k=num_group)
mean(mod_knn==dtst[[Y]])

# Confusion Matrix of the test set:
  
trn_res <- confusionMatrix(predict(mod_knn, dtrn[-1]), dtrn[[Y]]) 
trn_res$byClass[,c("Sensitivity", "Specificity", "Precision")]

tst_res <- confusionMatrix(predict(mod_knn, dtst[-1]), dtst[[Y]]) 
tst_res$byClass[,c("Sensitivity", "Specificity", "Precision")]


## svm

library(kernlab)

ctrl <- trainControl(method="cv",   
                    number=5,		    
                    classProbs = T)
grid <- expand.grid(sigma = c(.01, 0.0125, .015, 0.17, 0.2),
                    C = c(0.75, 0.9, 1, 1.1, 1.25))
                    
set.seed(seed_num)

svmRModel <- train(dtrn.proc[,-1], dtrn.proc[[Y]],
                  method = "svmRadial",
                  metric = "ROC",
                  preProc = c("center", "scale"),
                  tuneGrid = grid,
                  fit = FALSE,
                  trControl = ctrl)
                   
bst <- unlist(svmRModel$bestTune)
# Confusion Matrix of the test set:
  
trn_res <- confusionMatrix(predict(svmRModel, dtrn[-1]), dtrn[[Y]]) 
trn_res$byClass[,c("Sensitivity", "Specificity", "Precision")]

tst_res <- confusionMatrix(predict(svmRModel, dtst[-1]), dtst[[Y]]) 
tst_res$byClass[,c("Sensitivity", "Specificity", "Precision")]









