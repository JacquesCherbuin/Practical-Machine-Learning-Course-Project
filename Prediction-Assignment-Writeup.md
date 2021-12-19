## 1. Introduction

In this report we will create a machine learning algorithm to predict
personal activity. First, we will load and prepare the data. Second, we
will test different models and optimise them on the training and
validation sets. Finally, we will show the results of the model building
process and use the best model to predict on the test set.

## 2. Data Preprocessing

### 2.1 Load Packages

We load the required packages:

    library(caret)

    ## Loading required package: ggplot2

    ## Loading required package: lattice

    library(RColorBrewer)
    library(doParallel)

    ## Loading required package: foreach

    ## Loading required package: iterators

    ## Loading required package: parallel

    cl <- makePSOCKcluster(4)
    registerDoParallel(cl)

### 2.2 Set Seed

We set the seed for reproducibility:

    set.seed(123)

### 2.3 Load Data

We download the data and store it in training and test variables:

    url_training <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    url_test <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    training <- read.csv(url(url_training), na.strings=c("NA","#DIV/0!",""))
    testing <- read.csv(url(url_test), na.strings=c("NA","#DIV/0!",""))

### 2.3 Clean Data

Remove columns with NA values and irrelevant variables:

    training <- training[,colSums(is.na(training))==0]
    training <- training[,-(1:7)]
    testing <- testing[,colSums(is.na(testing))==0]
    testing <- testing[,-(1:7)]

### 2.4 Split into Training and Validation Sets

Split training set into training set and validation set to estimate out
of sample error:

    in_training <- createDataPartition(training$classe,p=0.7, list = FALSE)
    validation <- training[-in_training,]
    training <- training[in_training,]

## 3. Model Building

We will build 3 different models and then we will use the validation set
to choose the best performing model.

### 3.1 Random Forest

We create the random forest model, predict on the validation set, and
get the confusion matrix to judge accuracy:

    rf_model <- train(classe~., preProcess=c('center','scale'), data = training,method="rf")
    rf_prediction <- predict(rf_model, validation)
    rf_confusion <- confusionMatrix(rf_prediction, factor(validation$classe))

### 3.2 Boosted Gradient Machine

We create the boosted gradient machine model, predict on the validation
set, and get the confusion matrix to judge accuracy:

    gbm_model <- train(classe~., preProcess=c('center','scale'), data = training,method="gbm", verbose = FALSE)
    gbm_prediction <- predict(gbm_model, validation)
    gbm_confusion <- confusionMatrix(gbm_prediction, factor(validation$classe))

### 3.3 Linear Disciriminant Analysis

We create the linear discriminant analysis model, predict on the
validation set, and get the confusion matrix to judge accuracy:

    lda_model <- train(classe~.,preProcess=c('center','scale'), data = training,method="lda")
    lda_prediction <- predict(lda_model, validation)
    lda_confusion <- confusionMatrix(lda_prediction, factor(validation$classe))

## 4. Results

We show the results of our 3 models on the validation data:

    paste("Random Forest Accuracy:            ", round(rf_confusion$overall['Accuracy'],5))

    ## [1] "Random Forest Accuracy:             0.99269"

    paste("Gradient Boosted Machine Accuracy: ", round(gbm_confusion$overall['Accuracy'],5))

    ## [1] "Gradient Boosted Machine Accuracy:  0.9616"

    paste("Linear Disciriminant Analysis Accuracy:   ", round(lda_confusion$overall['Accuracy'],5))

    ## [1] "Linear Disciriminant Analysis Accuracy:    0.69601"

We plot the accuracy of our three models on the validation data:

    accuracy <- c(rf_confusion$overall['Accuracy'],gbm_confusion$overall['Accuracy'],lda_confusion$overall['Accuracy'])
    par(mar=c(4,6,3,3), cex=0.8)
    barplot(accuracy, names.arg =c('Random Forest', 'Gradient Boosted Machine', 'Linear Discriminant Analysis'), main = 'Model Accuracy',ylab = "Accuracy", col=brewer.pal(9,"Greens")[3:9])

![](Prediction-Assignment-Writeup_files/figure-markdown_strict/unnamed-chunk-10-1.png)

The model with the highest accuracy is the random forest model. We will
choose this model and apply it to the test set:

    rf_testing_prediction <- predict(rf_model, testing)
    par(mar=c(4,6,3,3), cex=0.8)
    plot(rf_testing_prediction, col=brewer.pal(9,'Blues')[3:9], main = 'Final Model Predictions',ylab = 'Frequency')

![](Prediction-Assignment-Writeup_files/figure-markdown_strict/unnamed-chunk-11-1.png)
