# Finance-Research

---
    International Stock Market Prediction using Artificial Neural Networks\
    AC 299r Independent Study 
---

## Motivation

A body of literature in financial economics suggests that international equity markets can have cross-market momentum, where one market in a region correlates with another market in that region (or possibly in a different region) in a lagged manner. Profitable trading strategies are devised to exploit the predictability of future prices using cross market momentum. 

Previous studies primarily focus on the dynamics among selected major country indices such as the US, UK, Germany, and China, while ignoring the dynamics with other countries in the same region or across regions. In this study, we seek to investigate the price predictability of the international equity indices by looking at regional indices as well as major country indices that cover the world equity markets according to the MSCI world classification. We use large, liquid iShares Exchange-Traded-Funds (ETF) and SPDR S&P 500 ETF data downloaded from Yahoo! Finance.

## Problem Statement

 we break daily returns of a region or country into intraday and overnight returns that are driven by fundamentally different drivers, depending on the overlapping time zone.


Given the past cross-sectional price information of equity indices, our goal is to predict the next-day overnight or intraday return of one index:

$$ \hat R_{overnight, t+1} = f ( R_{intraday,t}, C_{t} , \text{Technical indicators}_{t}) $$
 
$$ \hat R_{intraday,t+1} =f( R_{overnight, t+1}  , O_{ t+1}  ,\text{Technical indicators}_{t})$$

<center><img src="problem.png" align="middle" style="width: 400px;"/></center>

\noindent where $R_t$ denotes cross-sectional intraday or overnight returns of day $t$; $O_t,C_t$ denotes the cross-sectional last available open or close price of day $t$. On any day $t$, overnight returns come first, then come intraday and daily returns. Technical indicators of day $t$ are calculated based on last available close price of the index we are predicting (see the feature engineering section). $f$ is a model that generates a prediction given the inputs. 

## Feature Engineering 
The following momentum indicators are selected from the literature [nikkei, chen] that are reported to have the more predictive power than other technical indicators. They are all based on price and volume information at or before time t:
1.     Exponential and Simple Moving Averages over k-period rolling window
2.     Past k-period volatility
3.     Cross-sectional Price Trend indicators 
AD: number of advancing stocks at time t minus that of declining stocks
ADV: volume of advancing stocks at time t minus that of declining stocks
4.     Past k-period change in price
5.     Past k-period log return
6.    Past k-period stochastic oscillator %K and stochastic oscillator % D
The parameter k is selected by the author within a range of 5 days for all indicators. For exponential moving averages, k = 1,2,3...,10,15,20,25. 


## Neural Network Architecture
A 2-layer fully-connected network: 20 input units - 24 hidden units - 3 hidden units - 1 output unit
Mean Square Loss function
ADAM optimizer 
ReLu activation for all nodes except linear activation for output node
L2 penalty; max norm constraints of weights; dropout layers
Monitor training process – early stopping if validation not improving

Ensemble Forecasting Method
We break the time series into multiple rolling windows of training-validation-test sets. For each window, we train 100 neural networks on the training set and choose the top 50% models by accuracy rates on the validation set. They form a committee and output an average prediction as the final decision.


<center><img src="pipeline.png" align="middle" style="width: 400px;"/></center>


## Experimental Results
We test our models from 2015-09-08 to 2017-04-07 over 400 market days for Asia ex Japan which fewer data, and from 2015-02-05 to 2017-04-07 over 800 market days for all other regions. Results are shown in Figures \ref{plot1}, \ref{plot2}, \ref{plot3}, and \ref{plot4}. For comparison, a baseline is calculated as the fraction of positive returns in the test set, which does not vary with the proportion of transaction $100 \cdot (1-p)\%$. 


<center><img src="intraday_final1.png" align="middle" style="width: 400px;"/></center>

<center><img src="intraday_final2.png" align="middle" style="width: 400px;"/></center>

<center><img src="overnight_final1.png" align="middle" style="width: 400px;"/></center>

<center><img src="overnight_final2.png" align="middle" style="width: 400px;"/></center>


We can make several observations:
-   For overnight returns, on average, all the models perform slightly
    better than the baseline, except for Asia ex Japan. For intraday
    returns, on average, the models perform on par with the baselines.
    This suggests that there is predictive information in the features
    of overnight returns across all regions. One reason might be that
    the trading outside the exchange hours does not happen as often as
    within them, so that the market does not react as timely as
    information arrives. In addition, in a less efficient after-hour
    market, the bid-ask spreads are probably higher in trading;
    therefore, to access the execution feasibility of the trading
    strategy, it would be helpful to compute the break-even trading
    costs based on our model returns to compare with real-world trading
    costs.

-   A prediction accuracy higher than the baseline is not always
    equivalent to a good Sharpe ratio or feasible strategy; nor does a
    feasible strategy require a high prediction accuracy. For example,
    our ensemble neural net has below baseline accuracy in predicting
    overnight returns in Asia ex Japan. Yet, its Sharpe ratio is
    significantly above 1.0 and outperforms the benchmark models. This
    suggests a small proportion of good predictions that turn out to
    earn large returns can outweigh a large proportion of bad
    predictions with smaller losses. On the other hand, the benchmark
    models with higher-than-baseline accuracy, for example in predicting
    Canada overnight returns, can have zero or even negative Sharpe
    ratios. Therefore, it is important to observe consistency in both
    accuracy and Sharpe ratio curves to confirm performance stability of
    a model.

-   In terms of Sharpe ratio, our ensemble neural net outperforms the
    benchmarks in predicting intraday returns in China and overnight
    returns in Asia ex Japan. In all other cases, its performance is on
    average at par or even worse than the benchmark models. The sub-par
    performance highlights one potential issue with the neural net that
    the ensemble after model averaging can still have variation that
    cause performance instability, because the neural nets with powerful
    learning capacity can mistake noises as true patterns in the data.
    However, given the large set of neural network hyper-parameters
    (precise architecture, learning rate, convergence criteria,
    regularization, etc), we have minimal amount of fine-tuning the
    algorithm, striking a balance between an overly-tuned model and
    simplicity. It is certainly possible to select optimal parameters
    using other machine learning algorithms such as particle swarm
    optimization and genetic algorithm. Another explanation for the
    performance instability is that we have given very similar input
    features for all experiments; however, returns of different types
    and regions may have different dynamics and require specialized
    features.

## Concluding Remarks
==================

In this study, we propose to predict the intraday and overnight returns
of international equity indices that cover most of the world equity
markets. To minimize spurious correlations due to low liquid or
asynchronous trading, we use large, liquid Exchange-Traded-Funds trade
data that closely track MSCI regional indexes or the S&P 500. We aim to
predict the next intraday or overnight returns of an ETF based on the
last available price and volume information. We computed the technical
indicators of the price and volume information and selected the most
important features as inputs. Then we train an ensemble of neural
networks and benchmark it with Lasso and Ridge, in terms of adjusted
accuracy rates and Sharpe ratio. The results show that in all regions
except for Asia ex Japan, all the models outperform the baseline in
predicting overnight returns in terms of directional accuracy, which
suggests predictability of cross-market momentum. However, quantifying
to what extent the result can be attributed to thin markets for
overnight returns is outside the scope of this study.

In addition, our ensemble of 2-layer models is on average on par with
the regularized linear models as benchmarks. This highlight some issues
regarding application of neural networks to financial time series: 1)
much of the nuisance the neural net learns (better than a simple linear
model) in financial time series can be just noise, which is random or
inherently unpredictable. 2) to fine-tuning an ensemble to beat linear
methods, the computational cost and risks of over-fitting are high; and
3) to improve the overall performance, no matter what method to use, one
could focus on engineering the appropriate features in addition to
tuning the model.


