Homework 1
================
Painter, Ty
Fri Feb 5 11:23:42 2021

**Using the RMarkdown/knitr/github mechanism, implement the following
tasks by extending the example R script mixture-data-lin-knn.R:**

  - Paste the code from the mixture-data-lin-knn.R file into the
    homework template Knitr document.
  - Read the help file for R’s built-in linear regression function lm

<!-- end list -->

``` r
library('class') # has knn fucntion
library('dplyr') # data manipulation...tidyverse
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
## load binary classification example data from author website 
## 'ElemStatLearn' package no longer available
load(url('https://web.stanford.edu/~hastie/ElemStatLearn/datasets/ESL.mixture.rda'))
dat <- ESL.mixture # list

plot_mix_data <- expression({ # expression = piece of code not evaluated; could write as a function
  plot(dat$x[,1], dat$x[,2], # plot plots x1 on x-axis and x2 on y-axis
       col=ifelse(dat$y==0, 'blue', 'orange'),
       pch=20, 
       xlab=expression(x[1]),
       ylab=expression(x[2]))
  ## draw Bayes (True) classification boundary
  prob <- matrix(dat$prob, length(dat$px1), length(dat$px2)) # dat$prob = prob of orange
  cont <- contourLines(dat$px1, dat$px2, prob, levels=0.5) # countour of certain value of prob (=0.5), can adjust levels to whatever prob we want
  rslt <- sapply(cont, lines, col='purple') # plots contour line, farther away from the 0.5 countour line is higher prob
})

eval(plot_mix_data) # evaluates expression above
```

![](hw1_files/figure-gfm/unnamed-chunk-1-1.png)<!-- -->

  - Re-write the functions fit\_lc and predict\_lc using lm, and the
    associated predict method for lm objects.

<!-- end list -->

``` r
## fit linear classifier with lm() function, estimates beta
fit_lc2 <- function(y,x1, x2) {
  #as.vector(summary(lm(y~x1+x2))$coefficients[,1])
  lm(y~x1+x2)
}
```

  - Consider making the linear classifier more flexible, by adding
    squared terms for x1 and x2 to the linear model

<!-- end list -->

``` r
lc_beta2 <- fit_lc2(dat$y, dat$x[,1]^2, dat$x[,2]^2) # square x1 and x2
lc_pred2 <- predict(lc_beta2, data.frame(dat$xnew)) # predict values for all combinations

## reshape predictions as a matrix
lc_pred2 <- matrix(lc_pred2, length(dat$px1), length(dat$px2)) # surface predicted by linear classifier model
contour(lc_pred2, # 0.5 show decision line, y-hat surface
      xlab=expression(x[1]), # hard to use linear classifier when prob can be >1 and <0
      ylab=expression(x[2]))
```

![](hw1_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

``` r
## find the contours in 2D space such that lc_pred == 0.5
lc_cont2 <- contourLines(dat$px1, dat$px2, lc_pred2, levels=0.5)

## plot data and decision surface
eval(plot_mix_data) # 0.5 contour line
sapply(lc_cont2, lines) # linear decision line
```

![](hw1_files/figure-gfm/unnamed-chunk-3-2.png)<!-- -->

    ## [[1]]
    ## NULL

  - Describe how this more flexible model affects the bias-variance
    tradeoff
    
    Squaring both predictor values (x1 and x2) caused the decision line
    to become more biased but also lowered the variance. In the contour
    plot the values only range from 0.15-0.65 which is smaller range
    than the original indicating a smaller variance. With this decrease
    in variance the bias increases since the 0.5 line increases along
    the x2 axis. This movement is based on assumptions that the x1 and
    x2 relationship is linear, which it is not based on the shape of the
    purple Bayes decision boundry.
