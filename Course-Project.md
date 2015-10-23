---
title: "Practical Machine Learning Course project"
author: "Sabin Khadka"
date: "October 19, 2015"
output: html_document
---

##Introduction
In this project we use machine learning appraoch to classify activities according to the data collected from acceleromters of the belt, forearm, arm and dumbell of six male participants. The participants are asked to lift barbell in 5 different ways (1 correctly and 4 incorrectly). Class A: exactly according to the specification; class B: throwing the elbows to the front; Class C: lifting the dumbbell only halfway; Class D: lowering the dumbbell only halfway; Class E: throwing the hips to the front. More detailed explanation of the experiment and dataset are provided in Vellosso, E. et al. 2013. Refer to http://groupware.les.inf.puc-rio.br/har for more details. Here, in the current analysis we'll be using random forest tree and caret library functions to classify the activities. 

The training and testing datasets are downloaded from https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv & https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv respectively.



```r
# Load caret and randomForest libraries in R
wd <- getwd() # working directory
library("caret")
library("randomForest")
# Set seed for reproducibility
set.seed(12345)
#If running for the first time create Project folder
dir.create("Project", showWarnings=FALSE)
cat("source URL for training and testing data")
```

```
## source URL for training and testing data
```

```r
# URL for training and testing datasets
trainURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testURL  <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists("./Project/pml-training.csv")) {
        download.file(trainURL, destfile = "./Project/pml-training.csv", method = "curl")  
}
if (!file.exists("./Project/pml-testing.csv")) {
        download.file(testURL, destfile = "./Project/pml-testing.csv", method = "curl") 
}
trn_df <- read.csv("./Project/pml-training.csv", header =T, na.strings=c("NA",""))
tst_df <- read.csv("./Project/pml-testing.csv", header =T, na.strings=c("NA",""))
```

##Data exploration

First we'll check the proportion of missing data. We'll only consider the predictors containing 75% or more non-missing values. We'll remove all other predictors so as to maximize the number of observations.


```r
na_count <-data.frame(sapply(trn_df, function(y) sum(length(which(is.na(y))))))
na_prop <- na_count/dim(trn_df)[1]
trn_df1 <- trn_df[,!(na_prop>0.75)] # Remove columns with less than 75% of the valid data in training set
tst_df1 <- tst_df[,!(na_prop>0.75)] # Remove columns with less than 75% of the valid data in training set
```

Checking if new cleaned training dataset has NA values. Does training dataset contains NA's? FALSE

Checking if new cleaned testing dataset has NA values. Does training dataset contains NA's? FALSE

The remaining predictors are:


```r
colnames(trn_df1)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
##  [7] "num_window"           "roll_belt"            "pitch_belt"          
## [10] "yaw_belt"             "total_accel_belt"     "gyros_belt_x"        
## [13] "gyros_belt_y"         "gyros_belt_z"         "accel_belt_x"        
## [16] "accel_belt_y"         "accel_belt_z"         "magnet_belt_x"       
## [19] "magnet_belt_y"        "magnet_belt_z"        "roll_arm"            
## [22] "pitch_arm"            "yaw_arm"              "total_accel_arm"     
## [25] "gyros_arm_x"          "gyros_arm_y"          "gyros_arm_z"         
## [28] "accel_arm_x"          "accel_arm_y"          "accel_arm_z"         
## [31] "magnet_arm_x"         "magnet_arm_y"         "magnet_arm_z"        
## [34] "roll_dumbbell"        "pitch_dumbbell"       "yaw_dumbbell"        
## [37] "total_accel_dumbbell" "gyros_dumbbell_x"     "gyros_dumbbell_y"    
## [40] "gyros_dumbbell_z"     "accel_dumbbell_x"     "accel_dumbbell_y"    
## [43] "accel_dumbbell_z"     "magnet_dumbbell_x"    "magnet_dumbbell_y"   
## [46] "magnet_dumbbell_z"    "roll_forearm"         "pitch_forearm"       
## [49] "yaw_forearm"          "total_accel_forearm"  "gyros_forearm_x"     
## [52] "gyros_forearm_y"      "gyros_forearm_z"      "accel_forearm_x"     
## [55] "accel_forearm_y"      "accel_forearm_z"      "magnet_forearm_x"    
## [58] "magnet_forearm_y"     "magnet_forearm_z"     "classe"
```

We'll remove the remove column that cannot be used as predictors (columns 1:7). Also, we'll check if any of the predictors are uninformative using near zero variance in caret. We'll remove any uniformative predictors that are constant (near zero variance) across the sample. 


```r
trn_df2 <- trn_df1[,c(8:dim(trn_df1)[2])]
tst_df2 <- tst_df1[,c(8:dim(tst_df1)[2])]
nZroVar <- nearZeroVar(trn_df2, saveMetrics = T) # Check for near zero variance predictors
trn_df2 <- trn_df2[,nZroVar$nzv==FALSE]
tst_df2 <- tst_df2[,nZroVar$nzv==FALSE]
```

List of final predictors:


```r
colnames(trn_df2[-length(trn_df2)])
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"
```

Does training data have any predictors with near zero variance (TRUE=same; FALSE=losing some predictors)? TRUE

Next, we'll randomly split the training dataset set into train/test dataset into 60/40 prorportion. For this purpose we'll use createDataPartition function form caret. 


```r
inTrain <- createDataPartition(trn_df2$classe, p = 0.6, list =F, times =1)
trn_train <- trn_df2[inTrain,]
trn_test  <-trn_df2[-inTrain,]
```

Number of observations in training dataset: 11776

Number of observations in testing dataset: 7846

Building model for classification. We'll use random forest technique.


```r
# Random Forest takes very very long to run. Saved the model so as to no need to run next time. 
if(!file.exists("./Project/RForestModel1.Rdata")) {
        model1 <- train(classe ~ ., data = trn_train, method = "rf")
        save(model1, "./Project/RForestModel1.Rdata")
} else {
        load("./Project/RForestModel1.Rdata")
}
# Print the model
print(model1$finalModel)
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.9%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3338    8    1    0    1 0.002986858
## B   21 2247   10    1    0 0.014041246
## C    0   13 2032    9    0 0.010710808
## D    0    0   27 1900    3 0.015544041
## E    0    2    3    7 2153 0.005542725
```

For cross-validation we'll use the 40% of the splitted dataset from training set. The confusion matrix shows the expected and predicted classe of the cross-validation dataset, Accuracy and overall statistis of the model.



```r
predict_tst <- predict(model1, newdata = trn_test) 
ConfMat <- confusionMatrix(predict_tst, trn_test$classe)
# Print Confusion matrix
ConfMat
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    3    0    0    0
##          B    0 1509    1    0    1
##          C    0    6 1366    9    3
##          D    0    0    1 1276    2
##          E    0    0    0    1 1436
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9966         
##                  95% CI : (0.995, 0.9977)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9956         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9941   0.9985   0.9922   0.9958
## Specificity            0.9995   0.9997   0.9972   0.9995   0.9998
## Pos Pred Value         0.9987   0.9987   0.9870   0.9977   0.9993
## Neg Pred Value         1.0000   0.9986   0.9997   0.9985   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1923   0.1741   0.1626   0.1830
## Detection Prevalence   0.2849   0.1926   0.1764   0.1630   0.1832
## Balanced Accuracy      0.9997   0.9969   0.9979   0.9959   0.9978
```

The out of sample error is `1-ConfMat[3]$overall[1]`. We can conclude that the model did performed well to classify different activities.

Predict and create test data submission files.


```r
predict_testSet <- predict(model1, newdata = tst_df2) 
pml_write_files = function(x) {
        n=length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i], file=filename, quote=FALSE, row.names=FALSE, col.names=FALSE)
        }
}
setwd(paste0(wd,"/Project"))
pml_write_files(predict_testSet)
setwd(wd)
```
Reference. Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. Qualitative Activity Recognition of Weight Lifting Exercises. Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
