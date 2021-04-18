Homework 6
================
Painter, Ty
Sun Apr 18 10:30:35 2021

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

tune_params <- as.data.frame(rf_fit[4]) # make grid search accuracy a data frame
tune_params <- tune_params %>% arrange(desc(results.Accuracy)) # sort by accuracy and select the params with best accuracy
tune_params[1,]
```

    ##   results.mtry results.min.node.size results.splitrule results.Accuracy
    ## 1            3                     1              gini        0.9659939
    ##   results.Kappa results.AccuracySD results.KappaSD
    ## 1     0.9625863         0.02524326      0.02777005

I created a data frame with the results from the grid search and
arranged the results in order of highest accuracy and selected the top
row. The parameters with the best accuracy are 3 variables sampled at
each split and a minimum of 1 terminal node.

``` r
set.seed(12)
rf_best_fit <- randomForest(y ~ ., 
                            data=train,
                            nodesize=1,
                            mtry=3)
```

I created a best fit model with the specified best parameters.

### 5\. With the tuned model, make predictions using the majority vote method, and compute the misclassification rate using the ‘vowel.test’ data.

``` r
test <- read.csv(url("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/vowel.test"))
test <- test[,2:ncol(test)]
test_y <- as.factor(test$y)
## compute test error on testing data
test_preds <- predict(rf_best_fit, newdata = test)
cverr <- rep(NA, length(test))

confusionMatrix(test_y, test_preds) 
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction  1  2  3  4  5  6  7  8  9 10 11
    ##         1  34  6  2  0  0  0  0  0  0  0  0
    ##         2   0 24 14  0  0  0  0  0  4  0  0
    ##         3   0  3 27  5  0  4  0  0  0  0  3
    ##         4   0  0  3 30  0  9  0  0  0  0  0
    ##         5   0  0  0  3 21 13  4  0  0  0  1
    ##         6   0  0  0  0 11 21  0  0  0  0 10
    ##         7   0  0  0  0 10  1 26  0  5  0  0
    ##         8   0  0  0  0  0  0  6 29  7  0  0
    ##         9   0  0  0  0  0  0  5  6 24  1  6
    ##         10  5 14  0  0  0  0  0  1  1 21  0
    ##         11  0  1  0  1  0  5  4  0 12  0 19
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.5974          
    ##                  95% CI : (0.5511, 0.6425)
    ##     No Information Rate : 0.1147          
    ##     P-Value [Acc > NIR] : < 2.2e-16       
    ##                                           
    ##                   Kappa : 0.5571          
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: 1 Class: 2 Class: 3 Class: 4 Class: 5 Class: 6
    ## Sensitivity           0.87179  0.50000  0.58696  0.76923  0.50000  0.39623
    ## Specificity           0.98109  0.95652  0.96394  0.97163  0.95000  0.94866
    ## Pos Pred Value        0.80952  0.57143  0.64286  0.71429  0.50000  0.50000
    ## Neg Pred Value        0.98810  0.94286  0.95476  0.97857  0.95000  0.92381
    ## Prevalence            0.08442  0.10390  0.09957  0.08442  0.09091  0.11472
    ## Detection Rate        0.07359  0.05195  0.05844  0.06494  0.04545  0.04545
    ## Detection Prevalence  0.09091  0.09091  0.09091  0.09091  0.09091  0.09091
    ## Balanced Accuracy     0.92644  0.72826  0.77545  0.87043  0.72500  0.67244
    ##                      Class: 7 Class: 8 Class: 9 Class: 10 Class: 11
    ## Sensitivity           0.57778  0.80556  0.45283   0.95455   0.48718
    ## Specificity           0.96163  0.96948  0.95599   0.95227   0.94563
    ## Pos Pred Value        0.61905  0.69048  0.57143   0.50000   0.45238
    ## Neg Pred Value        0.95476  0.98333  0.93095   0.99762   0.95238
    ## Prevalence            0.09740  0.07792  0.11472   0.04762   0.08442
    ## Detection Rate        0.05628  0.06277  0.05195   0.04545   0.04113
    ## Detection Prevalence  0.09091  0.09091  0.09091   0.09091   0.09091
    ## Balanced Accuracy     0.76970  0.88752  0.70441   0.95341   0.71640

The misclassification rate is \~40%.

``` r
# misclassification rate
accuracy <- as.data.frame(confusionMatrix(test_y, test_preds)[3])[1,]
1 - accuracy
```

    ## [1] 0.4025974
