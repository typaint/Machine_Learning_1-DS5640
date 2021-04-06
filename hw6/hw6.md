Homework 6
================
Painter, Ty
Mon Apr 5 22:33:56 2021

Goal: Understand and implement a random forest classifier.

Using the “vowel.train” data, and the “randomForest” function in the R
package “randomForest”. Develop a random forest classifier for the vowel
data by doing the following:

``` r
library(randomForest)
```

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

``` r
library(tidyverse)
```

    ## ── Attaching packages ─────────────────────────────────────── tidyverse 1.3.0 ──

    ## ✓ ggplot2 3.3.2     ✓ purrr   0.3.4
    ## ✓ tibble  3.0.6     ✓ dplyr   1.0.3
    ## ✓ tidyr   1.1.2     ✓ stringr 1.4.0
    ## ✓ readr   1.4.0     ✓ forcats 0.5.0

    ## ── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::combine()  masks randomForest::combine()
    ## x dplyr::filter()   masks stats::filter()
    ## x dplyr::lag()      masks stats::lag()
    ## x ggplot2::margin() masks randomForest::margin()

``` r
library(caret)
```

    ## Loading required package: lattice

    ## 
    ## Attaching package: 'caret'

    ## The following object is masked from 'package:purrr':
    ## 
    ##     lift

``` r
library(heuristica)
train <- read.csv(url("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.train"))
train <- train[,2:ncol(train)]
```

### 1\. Convert the response variable in the “vowel.train” data frame to a factor variable prior to training, so that “randomForest” does classification rather than regression.

``` r
train$y <- as.factor(train$y)
```

### 2\. Review the documentation for the “randomForest” function.

### 3\. Fit the random forest model to the vowel data using all of the 11 features using the default values of the tuning parameters.

``` r
fit <- randomForest(y ~ ., data=train)
```

### 4\. Use 5-fold CV and tune the model by performing a grid search for the following tuning parameters:

1)  the number of variables randomly sampled as candidates at each
    split; consider values 3, 4, and 5.
2)  the minimum size of terminal nodes; consider a sequence (1, 5, 10,
    20, 40, and 80).

<!-- end list -->

``` r
samp_vars <- c(3,4,5)
node_size <- c(1, 5, 10, 20, 40, 80)

# 5 folds 
control <- trainControl(method='cv', 
                        number=5,
                        search="grid")
set.seed(41)
# grid search on mtry and nodesize
tunegrid <- expand.grid(.mtry = samp_vars, 
                        .min.node.size = node_size, 
                        .splitrule = "gini")
rf_fit <- train(y~., 
                data = train, 
                trControl = control,
                metric = "Accuracy",
                method = 'ranger', 
                tuneGrid = tunegrid, 
                classification = TRUE)
```

### 5\. With the tuned model, make predictions using the majority vote method, and compute the misclassification rate using the ‘vowel.test’ data.

``` r
test <- read.csv(url("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.test"))
test <- test[,2:ncol(test)]
test_y <- as.factor(test$y)
## compute test error on testing data
test_preds <- predict(rf_fit, newdata = test)
cverr <- rep(NA, length(test))

confusionMatrix(test_y, test_preds) 
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3  4  5  6  7  8  9 10 11
    ##         1  34  7  1  0  0  0  0  0  0  0  0
    ##         2   0 24 14  0  0  0  0  0  4  0  0
    ##         3   0  3 27  4  0  2  0  0  0  0  6
    ##         4   0  0  3 29  0  9  0  0  0  0  1
    ##         5   0  0  0  3 17 17  4  0  0  0  1
    ##         6   0  0  0  0  9 23  0  0  0  0 10
    ##         7   0  0  0  0  9  1 27  0  5  0  0
    ##         8   0  0  0  0  0  0  6 29  7  0  0
    ##         9   0  0  0  0  0  0  5  6 23  2  6
    ##         10  1 13  4  0  0  0  0  1  3 20  0
    ##         11  0  1  0  2  0  5  4  0 12  0 18
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5866          
    ##                  95% CI : (0.5402, 0.6319)
    ##     No Information Rate : 0.1234          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.5452          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity           0.97143  0.50000  0.55102  0.76316  0.48571  0.40351
    ## Specificity           0.98126  0.95652  0.96368  0.96934  0.94145  0.95309
    ## Pos Pred Value        0.80952  0.57143  0.64286  0.69048  0.40476  0.54762
    ## Neg Pred Value        0.99762  0.94286  0.94762  0.97857  0.95714  0.91905
    ## Prevalence            0.07576  0.10390  0.10606  0.08225  0.07576  0.12338
    ## Detection Rate        0.07359  0.05195  0.05844  0.06277  0.03680  0.04978
    ## Detection Prevalence  0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
    ## Balanced Accuracy     0.97635  0.72826  0.75735  0.86625  0.71358  0.67830
    ##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11
    ## Sensitivity           0.58696  0.80556  0.42593   0.90909   0.42857
    ## Specificity           0.96394  0.96948  0.95343   0.95000   0.94286
    ## Pos Pred Value        0.64286  0.69048  0.54762   0.47619   0.42857
    ## Neg Pred Value        0.95476  0.98333  0.92619   0.99524   0.94286
    ## Prevalence            0.09957  0.07792  0.11688   0.04762   0.09091
    ## Detection Rate        0.05844  0.06277  0.04978   0.04329   0.03896
    ## Detection Prevalence  0.09091  0.09091  0.09091   0.09091   0.09091
    ## Balanced Accuracy     0.77545  0.88752  0.68968   0.92955   0.68571
