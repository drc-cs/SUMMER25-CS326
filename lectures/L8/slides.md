---
title: CS 326
separator: <!--s-->
verticalSeparator: <!--v-->

theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.7em; left: 0; width: 60%; position: absolute;">

  # Introduction to Data Science Pipelines
  ## L8

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%;">

  <iframe src="https://lottie.host/embed/216f7dd1-8085-4fd6-8511-8538a27cfb4a/PghzHsvgN5.lottie" height = "100%" width = "100%"></iframe>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Welcome to CS 326
  ## Please check in and enter the provided code.

  </div>
  </div>
  <div class="c2" style="width: 50%; height: auto;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Enter Code" width="100%" height="100%" style="border-radius: 10px;"></iframe>
  </div>
</div>

<!--s-->

<div class="header-slide">

# Time Series Modeling

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

# Intro Poll
## On a scale of 1-5, how confident are you with the **time series** methods such as:

1. Seasonal Decomposition
2. Stationarity & Differencing
3. Autocorrelation
4. ARIMA

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

## Agenda

Today we're going to talk about time series analysis, specifically building an intution for forecasting models. We'll cover the following topics:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Understanding Time Series Data
- Seasonal Decomposition
- Stationarity & Differencing
- Autocorrelation

### Time Series Forecasting
- Moving Average
- Autoregressive Models
- ARIMA (Theory)
- ARIMA (Practice)

### Evaluation
- Walk-Forward Validation
- Evaluation Metrics

</div>
<div class="c2 col-centered" style = "width: 50%">

<div>
<img src="https://storage.googleapis.com/slide_assets/forecast.jpg" style="margin: 0 auto; display: block;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Lokad 2016</span>
</div>
</div>
</div>

<!--s-->

<div class="header-slide">

# Understanding Time Series Data

</div>

<!--s-->

## Understanding Time Series | Seasonal Decomposition

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; margin: 0; padding: 0;">

### Trend
The long-term movement of a time series. That represents the general direction in which the data is moving over time.

### Seasonality
The periodic fluctuations in a time series that occur at regular intervals. For example, sales data may exhibit seasonality if sales increase during the holiday season.

### Residuals
Noise in a time series that cannot be explained by the trend or seasonality.

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>

<img src="https://sthalles.github.io/assets/time-series-decomposition/complete-seasonality-plot-additive.png" width="400" style="margin: 0; padding: 0; display: block;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Thalles 2019</span>

</div>
</div>`
</div>


<!--s-->

## Understanding Time Series | Seasonal Decomposition

Seasonal Decomposition is a technique used to separate a time series into its trend, seasonal, and residual components. Seasonal decomposition can help identify patterns in the time series data and make it easier to model. It can be viewed as a form of feature engineering.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Additive Seasonal Decomposition
The seasonal component is added to the trend and residual components.

$$ X_t = T_t + S_t + R_t $$

### Multiplicative Seasonal Decomposition

The seasonal component is multiplied by the trend and residual components.
$$ X_t = T_t \times S_t \times R_t $$

</div>

<div class="c2" style = "width: 50%; margin-top: 8%;">
<div>
<img src="https://sthalles.github.io/assets/time-series-decomposition/complete-seasonality-plot-additive.png" width="400" style="margin: 0 auto; display: block;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Thalles 2019</span>
</div>
</div>
</div>

<!--s-->

## Understanding Time Series | Seasonal Decomposition


Seasonal Decomposition is a technique used to separate a time series into its trend, seasonal, and residual components. Seasonal decomposition can help identify patterns in the time series data and make it easier to model. It can be viewed as a form of feature engineering.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; height: 100%;">

### **Question**: Does this figure represent additive or multiplicative decomposition?

<img src="https://sthalles.github.io/assets/time-series-decomposition/complete-seasonality-plot-additive.png" width="400" style="margin: 0 auto; display: block;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Thalles 2019</span>

</div>

<div class="c2" style = "width: 50%; height: 100%;">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L8 | Q1" width="100%" height="100%" style="border-radius: 10px;"></iframe>
</div>

<!--s-->

## Understanding Time Series | Stationarity

A time series is said to be **stationary** if its statistical properties such as mean, variance, and autocorrelation do not change over time. Many forecasting methods assume that the time series is stationary. The **Augmented Dickey-Fuller Test (ADF)** is a statistical test that can be used to test for stationarity.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Strict Stationarity
The joint distribution of any subset of time series observations is independent of time. This is a strong assumption that is rarely met in practice.

### Trend Stationarity
The mean of the time series is constant over time. This is a weaker form of stationarity that is more commonly used in practice.

</div>
<div class="c2" style = "width: 50%">

<div>
<img src="https://upload.wikimedia.org/wikipedia/commons/e/e1/Stationarycomparison.png" style="margin: 0; display: block;">
<span style="font-size: 0.6em; padding-top: 0.5em; text-align: center; display: block; color: grey;">Wikipedia 2024</span>
</div>
</div>
</div>

<!--s-->

## Understanding Time Series | Differencing

**Differencing** is a technique used to make a time series **stationary** by computing the difference between consecutive observations. Differencing can help remove trends and seasonality from a time series.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

$$ Y_t' = Y_t - Y_{t-1} $$

Where:
- $Y_t$ is the observation at time $t$.
- $Y_t'$ is the differenced observation at time $t$.

</div>
<div class="c2" style = "width: 50%">

<img src="https://storage.googleapis.com/blogs-images-new/ciscoblogs/1/2020/03/0e3efdd8-differencing.png" width="400" style="margin: 0 auto; display: block; border-radius: 10px;">
<span style="font-size: 0.6em; text-align: center; display: block; color: grey;">Wise, 2020</span>

</div>
</div>

<!--s-->

## Understanding Time Series | Autocorrelation

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Autocorrelation
A measure of the correlation between a time series and a lagged version of itself. 

$$ \text{Corr}(X_t, X_{t-k}) $$


### Partial Autocorrelation
A measure of the correlation between a time series and a lagged version of itself, controlling for the values of the time series at all shorter lags.

$$ \text{Corr}(X_t, X_{t-k} | X_{t-1}, X_{t-2}, \ldots, X_{t-k+1}) $$

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>
<img src = "https://i.makeagif.com/media/3-17-2017/CYdNJ7.gif" width="100%" style="margin: 0 auto; display: block; border-radius: 10px;">
<span style="font-size: 0.5em; text-align: center; display: block; color: grey; padding-top: 0.5em;">@osama063, 2016</span>
</div>
</div>
</div>

<!--s-->

## Understanding Time Series | Autocorrelation

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Autocorrelation
A measure of the correlation between a time series and a lagged version of itself. 

$$ \text{Corr}(X_t, X_{t-k}) $$


### Partial Autocorrelation
A measure of the correlation between a time series and a lagged version of itself, controlling for the values of the time series at all shorter lags.

$$ \text{Corr}(X_t, X_{t-k} | X_{t-1}, X_{t-2}, \ldots, X_{t-k+1}) $$

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>
<img src="https://storage.googleapis.com/cs326-bucket/lecture_13/observed.png" width="100%" style="margin: 0 auto; display: block;">
<img src="https://storage.googleapis.com/cs326-bucket/lecture_13/auto2.png" width="100%" style="margin: 0 auto; display: block;">
</div>
</div>
</div>


<!--s-->

## Understanding Time Series | Checkpoint TLDR;

### Seasonal Decomposition
A technique used to separate a time series into its trend, seasonal, and residual components.

### Stationarity
A time series is said to be stationary if its basic properties do not change over time.

### Differencing
A technique used to make a time series stationary by computing the difference between consecutive observations.

### Autocorrelation
A measure of the correlation between a time series and a lagged version of itself. Partial autocorrelation controls for the values of the time series at all shorter lags.


<!--s-->

<div class="header-slide">

# Time Series Forecasting

</div>

<!--s-->

## Time Series Forecasting | Introduction

Time series forecasting is the process of predicting future values based on past observations. Time series forecasting is used in a wide range of applications, such as sales forecasting, weather forecasting, and stock price prediction. 

The **ARIMA** (Autoregressive Integrated Moving Average) model is a popular time series forecasting model that combines autoregressive, moving average, and differencing components.

Before we dive into ARIMA, let's first discuss two simpler time series forecasting models to build intuition for the components of ARIMA: **Moving Average (MA)** and **Autoregressive (AR)** Models.

<!--s-->

## Time Series Forecasting | Autoregressive Models

**Autoregressive Models (AR)**: A type of time series model that predicts future values based on past observations. The AR model is based on the assumption that the time series is a linear combination of its past values. It's primarily used to capture the periodic structure of the time series.

AR(1) $$ X_t = \phi_1 X_{t-1} + c + \epsilon_t $$

Where:

- $X_t$ is the observed value at time $t$.
- $\phi_1$ is a learnable parameter of the model.
- $c$ is a constant term (intercept).
- $\epsilon_t$ is the white noise at time $t$.

<!--s-->

## Time Series Forecasting | Autoregressive Models

**Autoregressive Models (AR)**: A type of time series model that predicts future values based on past observations. The AR model is based on the assumption that the time series is a linear combination of its past values. It's primarily used to capture the periodic structure of the time series.

AR(p) $$ X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + c + \epsilon_t $$

Where:

- $X_t$ is the observed value at time $t$.
- $p$ is the number of lag observations included in the model.
- $\phi_1, \phi_2, \ldots, \phi_p$ are the parameters of the model.
- $c$ is a constant term (intercept).
- $\epsilon_t$ is the white noise at time $t$.

<!--s-->

## Time Series Forecasting | Autoregressive Models

**Autoregressive Models (AR)**: A type of time series model that predicts future values based on past observations. The AR model is based on the assumption that the time series is a linear combination of its past values. It's primarily used for capturing the periodic structure of the time series.

$$ X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + c + \epsilon_t $$

<iframe width = "100%" height = "70%" src="https://storage.googleapis.com/cs326-bucket/lecture_13/ARIMA_1_2.html" title="scatter_plot"></iframe>

<!--s-->

## Time Series Forecasting | Moving Average

**Moving Average (MA) Models**: A type of time series model that predicts future values based on the past prediction errors. A MA model's primary utility is to smooth out noise and short-term discrepancies from the mean.

MA(1) $$ X_t = \theta_1 \epsilon_{t-1} + \mu + \epsilon_t$$

<div class = "col-wrapper" style="font-size: 0.8em;">
<div class="c1" style = "width: 50%">

Where: 

- $X_t$ is the observed value at time $t$.
- $\theta_1$ is a learnable parameter of the model.
- $\mu$ is the mean of the time series.
- $\epsilon_t$ is the white noise at time $t$.

</div>
<div class="c2" style = "width: 50%">

Example with a $\mu = 10 $ and $\theta_1 = 0.5$:

| t | $\widehat{X}_t$ | $\epsilon_t$ | $X_t$ |
|---|------------|--------------|-------|
| 1 | 10         | -2            | 8    |
| 2 | 9         | 1           | 10    |
| 3 | 10.5         | 0            | 10.5    |
| 4 | 10         | 2           | 12     |
| 5 | 11         | -1           | 10    |


</div>
</div>

<!--s-->

## Time Series Forecasting | Moving Average

**Moving Average (MA) Models**: A type of time series model that predicts future values based on the past prediction errors. A MA model's primary utility is to smooth out noise and short-term discrepancies from the mean.

MA(1) $$ X_t = \theta_1 \epsilon_{t-1} + \mu + \epsilon_t$$

<iframe width = "100%" height = "70%" src="https://storage.googleapis.com/cs326-bucket/lecture_13/MA2.html" title="scatter_plot";></iframe>

<!--s-->

## Time Series Forecasting | Moving Average

**Moving Average (MA) Models**: A type of time series model that predicts future values based on the past prediction errors. A MA model's primary utility is to smooth out noise and short-term discrepancies from the mean.

MA(q) $$ X_t = \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q} + \mu + \epsilon_t$$

Where: 

- $X_t$ is the observed value at time $t$.
- $q$ is the number of lag prediction errors included in the model.
- $\theta_1, \theta_2, \ldots, \theta_q$ are the learnable parameters.
- $\mu$ is the mean of the time series.
- $\epsilon_t$ is the white noise at time $t$.

<!--s-->

## Time Series Forecasting | ARMA

**Autoregressive Models with Moving Average (ARMA)**: A type of time series model that combines autoregressive and moving average components.

The ARMA model is defined as:

$$ X_t = \phi_1 X_{t-1} + \phi_2 X_{t-2} + \ldots + \phi_p X_{t-p} + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q} + c + \epsilon_t $$


Where:

- $X_t$ is the observed value at time $t$.
- $\phi_1, \phi_2, \ldots, \phi_p$ are the autoregressive parameters.
- $\theta_1, \theta_2, \ldots, \theta_q$ are the moving average parameters.
- $c$ is a constant term (intercept).
- $\epsilon_t$ is the white noise at time $t$.


<!--s-->

## Time Series Forecasting | ARIMA

**Autoregressive Integrated Moving Average (ARIMA)**: A type of time series model that combines autoregressive, moving average, and differencing components.

The ARIMA model is defined as: 

$$ y_t' = \phi_1 y_{t-1}' + \phi_2 y_{t-2}' + \ldots + \phi_p y_{t-p}' + \theta_1 \epsilon_{t-1} + \theta_2 \epsilon_{t-2} + \ldots + \theta_q \epsilon_{t-q} + c + \epsilon_t $$

Where:

- $y_t'$ is the differenced observation at time $t$.
- $\phi_1, \phi_2, \ldots, \phi_p$ are the autoregressive parameters.
- $\theta_1, \theta_2, \ldots, \theta_q$ are the moving average parameters.
- $c$ is a constant term (intercept).
- $\epsilon_t$ is the white noise at time $t$.

<!--s-->

## Time Series Forecasting | Practical ARIMA

ARIMA takes three parameters when fitting a model ($p$, $d$, $q$). 

<div style="font-size: 0.7em;">

| Parameter | Description | Estimation |
|-----------|-------------|------------|
| $p$ | The number of lag observations included in the model (lag order for autoregression). | Where there is a dropoff in Partial Autocorrelation Function (PACF) (with gradual decline in ACF). |
| $d$ | The number of times that the raw observations are differenced (degree of differencing). | Minimum amount of differencing required to achieve a significant Augmented Dickey-Fuller Test (ADF). |
| $q$ | The number of prediction errors included in the model (order of moving average). | Where there is a dropoff in the Autocorrelation Function (ACF) (with gradual decline in PACF). |

</div>


<!--s-->

## Time Series Forecasting | Practical ARIMA

ARIMA takes three parameters when fitting a model ($p$, $d$, $q$). 

<div style="font-size: 0.7em;">

| Parameter | Description | Estimation |
|-----------|-------------|------------|
| $p$ | The number of lag observations included in the model (lag order for autoregression). | Where there is a dropoff in Partial Autocorrelation Function (PACF) (with gradual decline in ACF). |
| $d$ | The number of times that the raw observations are differenced (degree of differencing). | Minimum amount of differencing required to achieve a significant Augmented Dickey-Fuller Test (ADF). |
| $q$ | The number of prediction errors included in the model (order of moving average). | Where there is a dropoff in the Autocorrelation Function (ACF) (with gradual decline in PACF). |

</div>

**Question**: What is a reasonable value of $p$ based on the following? 


<img src = "https://storage.googleapis.com/cs326-bucket/lecture_13/ar.png" width="60%" style="margin: 0 auto; display: block;">
<span style="font-size: 0.5em; text-align: center; display: block; color: grey; ">Spur Economics 2022</span>

<!--s-->

## Time Series Forecasting | Practical ARIMA

ARIMA takes three parameters when fitting a model ($p$, $d$, $q$). 

<div style="font-size: 0.7em;">

| Parameter | Description | Estimation |
|-----------|-------------|------------|
| $p$ | The number of lag observations included in the model (lag order for autoregression). | Where there is a dropoff in Partial Autocorrelation Function (PACF) (with gradual decline in ACF). |
| $d$ | The number of times that the raw observations are differenced (degree of differencing). | Minimum amount of differencing required to achieve a significant Augmented Dickey-Fuller Test (ADF). |
| $q$ | The number of prediction errors included in the model (order of moving average). | Where there is a dropoff in the Autocorrelation Function (ACF) (with gradual decline in PACF). |

</div>

**Question**: What is a reasonable value of $d$ based on the following? 


```python
import numpy as np
from statsmodels.tsa.stattools import adfuller

timeseries = ...
for d in range(0, 3):
    diffed = np.diff(timeseries, n=d)
    result = adfuller(diffed)
    print(f"ADF Statistic for d={d}: {result[0]} p-value: {result[1]}")
```
```text
ADF Statistic for d=0: -2.5 p-value: 0.1
ADF Statistic for d=1: -3.2 p-value: 0.04
ADF Statistic for d=2: -4.1 p-value: 0.01
``` 

<!--s-->

## Time Series Forecasting | Practical ARIMA

ARIMA takes three parameters when fitting a model ($p$, $d$, $q$). 

<div style="font-size: 0.7em;">

| Parameter | Description | Estimation |
|-----------|-------------|------------|
| $p$ | The number of lag observations included in the model (lag order for autoregression). | Where there is a dropoff in Partial Autocorrelation Function (PACF) (with gradual decline in ACF). |
| $d$ | The number of times that the raw observations are differenced (degree of differencing). | Minimum amount of differencing required to achieve a significant Augmented Dickey-Fuller Test (ADF). |
| $q$ | The number of prediction errors included in the model (order of moving average). | Where there is a dropoff in the Autocorrelation Function (ACF) (with gradual decline in PACF). |

</div>

**Question**: What is a reasonable value of $q$ based on the following? 


<img src = "https://storage.googleapis.com/cs326-bucket/lecture_13/ma.png" width="60%" style="margin: 0 auto; display: block;">
<span style="font-size: 0.5em; text-align: center; display: block; color: grey; ">Spur Economics 2022</span>

<!--s-->

<div class="header-slide">

# Forecast Evaluation

</div>

<!--s-->

## Walk-Forward Validation

In walk-forward validation, the model is trained on historical data and then used to make predictions on future data. The model is then retrained on the updated historical data and used to make predictions on the next future data point. This process is repeated until all future data points have been predicted.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Train / Validate Period
The historical data used to train and validate the time series model.

### Test Period
The future data used to evaluate the generalization performance of the time series model.

</div>
<div class="c2" style = "width: 50%">

<img src = "https://storage.googleapis.com/slide_assets/walk-forward.png" width="100%" style="margin: 0 auto; display: block;">
<span style="font-size: 0.5em; text-align: center; display: block; color: grey;">Peeratiyuth, 2018</span>

</div>
</div>

<!--s-->

## Walk-Forward Validation

In walk-forward validation, the model is trained on historical data and then used to make predictions on future data. The model is then retrained on the updated historical data and used to make predictions on the next future data point. This process is repeated until all future data points have been predicted.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Train / Validate Period
The historical data used to train and validate the time series model.

### Test Period
The future data used to evaluate the generalization performance of the time series model.

</div>
<div class="c2" style = "width: 50%">

<img src = "https://storage.googleapis.com/slide_assets/holdout-walk-forward.png" width="100%" style="margin: 0 auto; display: block;"> 
<span style="font-size: 0.5em; text-align: center; display: block; color: grey;">Karaman, 2005</span>

</div>
</div>


<!--s-->

## Evaluation Metrics

<div style = "font-size: 0.8em">

**Mean Absolute Error (MAE)**: The average of the absolute errors between the predicted and actual values.

$$ MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| $$

**Mean Squared Error (MSE)**: The average of the squared errors between the predicted and actual values.

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

**Root Mean Squared Error (RMSE)**: The square root of the average of the squared errors between the predicted and actual values.

$$ RMSE = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 } $$

**Mean Absolute Percentage Error (MAPE)**: The average of the absolute percentage errors between the predicted and actual values.

$$ MAPE = \frac{1}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\% $$

</div>

<!--s-->

<div class="header-slide">

# Wrapping Up

</div>

<!--s-->

## Summary | What We Covered

<div style="font-size: 0.75em;">

| Term | Description |
|------|-------------|
| **Seasonal Decomposition** | A technique used to separate a time series into its trend, seasonal, and residual components. |
| **Stationarity** | A time series is said to be stationary if its basic properties do not change over time. |
| **Differencing** | A technique used to make a time series stationary by computing the difference between consecutive observations. |
| **Autocorrelation** | A measure of the correlation between a time series and a lagged version of itself. Partial autocorrelation controls for the values of the time series at all shorter lags. |
| **ARIMA** | A type of time series model that combines autoregressive, moving average, and differencing components. |
| **Walk-Forward Validation** | A method for evaluating the generalization performance of a time series model. |
| **Evaluation Metrics** | Regression metrics used to evaluate the performance of a time series model. |
</div>

<!--s-->

## Summary | What We Didn't Cover

<div style="font-size: 0.75em;">

| Topic | Description |
|-------|-------------|
| **Seasonal ARIMA** | An [extension of ARIMA](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) that includes seasonal components. |
| **SARIMAX** | An [extension of ARIMA](https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html) that includes seasonal components and exogenous variables. |
| **Box-Jenkins Approach** | A [systematic method](https://en.wikipedia.org/wiki/Box%E2%80%93Jenkins_method) for identifying, estimating, and diagnosing ARIMA models. |
| **Maximum Likelihood Estimation** | A [method](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation) for estimating the parameters of a statistical model by maximizing the likelihood function. |
| **Prophet** | A [forecasting tool](https://facebook.github.io/prophet/) developed by Facebook that is designed for forecasting time series data with strong seasonal patterns. |
| **SOTA Models** | Models like [TiDE](https://arxiv.org/abs/2304.08424) and [TSMixer](https://arxiv.org/abs/2303.06053) are state-of-the-art models for time series forecasting (2023+). |

</div>
<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

# Exit Poll
## On a scale of 1-5, how confident are you with the **time series** methods such as:

1. Seasonal Decomposition
2. Stationarity & Differencing
3. Autocorrelation
4. ARIMA

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Storytelling & Ethics

</div>

<!--s-->

# Agenda

<div class = "col-wrapper" style = "font-size: 0.8em;">
<div class="c1" style = "width: 50%">

# **Part 1**: Storytelling
- Importance of storytelling in data science.
- What every good data story should include.
- What every good presentation should include.

# **Part 2**: Ethics of Data Science

- Consent for Data
- Privacy
- Examining Bias
- Who is Held Accountable?
- Radicalization and Misinformation

</div>
<div class="c2" style = "width: 50%; border-left: 1px solid black;">

# **Part 3**: Career Tips

- Effective Networking
- Monetizing Your Curiosity
- Building a Personal Brand

</div>
</div>

<!--s-->

<div class = "col-wrapper">
<div class="c1 col-centered">

<div>

# Storytelling

</div>

</div>

<div class="c2 col-centered" style="width: 100%;">

<iframe src="https://lottie.host/embed/be87d8ec-8c45-4e5f-8fa5-a7c51ba95111/AeEwvQMzRj.json" width="100%" height="100%"></iframe>

</div>
</div>

<!--s-->

## Storytelling | Importance

Data science is not just about the results. It is about the **story** that the data tells.

An often-overlooked aspect of data science is the ability to **communicate** and **convince** others of the results of your analysis.

<img src="https://justinsighting.com/wp-content/uploads/2016/05/housing-price-and-square-feet-predicted.jpg" style="margin: 0 auto; display: block; width: 50%;">
<p style="text-align: center; font-size: 0.6em; color: grey;">JustInsighting, 2024</p>

<!--s-->

## Storytelling | Every Data Story Must Include ...

1. Background of Problem

2. Statement of Assumptions

3. Motivation for Solving the Problem

4. Explanation of your Analysis

5. Declared Limitations & Future Improvements

<!--s-->

## Storytelling | Background of Problem

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### What is the problem you are trying to solve?

We are trying to predict the price of a house based on its square footage.

</div>
<div class="c2" style = "width: 50%">

<img src="https://justinsighting.com/wp-content/uploads/2016/05/housing-price-and-square-feet-predicted.jpg" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">JustInsighting, 2024</p>

</div>
</div>

<!--s-->

## Storytelling | Statement of Assumptions

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### What assumptions are you making in your analysis?

We assume that the data we are training on represents the general population.

### What are the implications of these assumptions?

If this assumption is incorrect, the model may fail to generalize.

</div>
<div class="c2" style = "width: 50%">
<img src="https://justinsighting.com/wp-content/uploads/2016/05/housing-price-and-square-feet-predicted.jpg" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">JustInsighting, 2024</p>

</div>
</div>


<!--s-->

## Storytelling | Motivation for Solving the Problem

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Why is it important to solve this problem?

Predicting the price of a house can help buyers and sellers make informed decisions.

</div>
<div class="c2" style = "width: 50%">

<img src="https://justinsighting.com/wp-content/uploads/2016/05/housing-price-and-square-feet-predicted.jpg" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">JustInsighting, 2024</p>

</div>
</div>

<!--s-->

## Storytelling | Explanation of your Analysis

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### How did you analyze the data?

We used linear regression to predict the price of a house based on its square footage.

### How do you interpret the results?

Our linear model predicts that the price of a house increases by **$100 per square foot**. Note that we don't report MSE or RMSE here.

</div>
<div class="c2" style = "width: 50%">

<img src="https://justinsighting.com/wp-content/uploads/2016/05/housing-price-and-square-feet-predicted.jpg" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">JustInsighting, 2024</p>

</div>
</div>

<!--s-->

## Storytelling | Declared Limitations & Future Improvements

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### What are the limitations of your analysis?

The model may not be accurate for houses with unique features, such as a swimming pool.

### How can you improve the analysis in the future?

You can collect more data on houses with swimming pools to improve the accuracy of the model.

</div>
<div class="c2" style = "width: 50%">

<img src="https://justinsighting.com/wp-content/uploads/2016/05/housing-price-and-square-feet-predicted.jpg" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">JustInsighting, 2024</p>

</div>
</div>

<!--s-->

## Storytelling Recap | Every Data Story Must Include ...

1. Background of Problem

2. Statement of Assumptions

3. Motivation for Solving the Problem

4. Explanation of your Analysis

5. Declared Limitations & Future Improvements

<!--s-->

## Storytelling | Every Good **Presentation** Must Include ...

1. Clear and concise slides

2. A compelling narrative

3. Energy and confidence 

<!--s-->

## Storytelling | Clear and Concise Slides

### What makes a slide **clear** and **concise**?

- Use bullet points to summarize key points.
- Use visuals to illustrate complex concepts.
- Use a consistent font and color scheme.

<!--s-->

## Storytelling | A Compelling Narrative

### What makes a narrative **compelling**?

- Tell a story that engages the audience.
- Use examples and anecdotes to illustrate key points.
- Use humor and emotion to connect with the audience.

<!--s-->

## Storytelling | Energy and Confidence

### How can you project **energy** and **confidence**?

- Speak clearly and with sufficient volume.
- Make eye contact with the audience.
- Use body language to emphasize key points.

<!--s-->

## Storytelling | Every Good **Presentation** Must Include ...

1. Clear and concise slides

2. A compelling narrative

3. Energy and confidence 

<!--s-->

<div class = "col-wrapper">
<div class="c1 col-centered" style="width: 60%;">

<div>

# Ethics of Data Science

</div>

</div>

<div class="c2 col-centered" style="width: 40%;">

<iframe src="https://lottie.host/embed/0059f4db-7136-402f-89a4-36b9230ca9aa/Jb6iMtHqzR.json" width="100%" height="100%"></iframe>

</div>
</div>

<!--s-->

## Ethics of Data Science | Topics

1. Consent for Data
2. Privacy
3. Examining Bias
4. Accountability
5. Radicalization and Misinformation

<!--s-->

## Ethics of Data Science | Consent for Data

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Why is consent important in data science?

- To protect the privacy of individuals.
- To ensure that data is used ethically and responsibly.

### How can you obtain consent for data?

- Inform individuals about how their data will be used.
- Obtain explicit consent before collecting or using data.

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>
<img src="https://www.euractiv.com/wp-content/uploads/sites/2/2024/02/shutterstock_1978079195-800x450.jpg" style="margin: 0 auto; display: block; width: 100%; border-radius: 5px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Euractiv, 2024</p>
</div>
</div>
</div>


<!--s-->

## Ethics of Data Science | Consent for Data

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### **Opt-in** vs. **Opt-out**

- Opt-in: Individuals must actively consent to the use of their data.
- Opt-out: Individuals must actively decline the use of their data.

### **Granular** vs. **Broad**

- Granular: Individuals can choose how their data is used.
- Broad: Individuals have limited control over how their data is used.

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>
<img src="https://www.euractiv.com/wp-content/uploads/sites/2/2024/02/shutterstock_1978079195-800x450.jpg" style="margin: 0 auto; display: block; width: 100%; border-radius: 5px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Euractiv, 2024</p>
</div>
</div>
</div>

<!--s-->

## Ethics of Data Science | Privacy


<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Why is privacy important in data science?

- To protect the personal information of individuals.
- To prevent the misuse of data for malicious purposes.

### How can you protect privacy in data science?

- Anonymize data to remove personally identifiable information.
- Encrypt data to prevent unauthorized access.
- Limit access to data to authorized individuals.

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>
<img src="https://images.theconversation.com/files/218904/original/file-20180514-100697-ig8lqn.jpg?ixlib=rb-4.1.0&q=45&auto=format&w=926&fit=clip" style="margin: 0 auto; display: block; width: 100%; border-radius: 5px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">SBPhotos, 2018</p>
</div>
</div>
</div>

<!--s-->

## Ethics of Data Science | Privacy Compliance with Regulations

<div style="font-size: 0.8em;">

| Regulation | Description |
|------------|-------------|
| **General Data Protection Regulation (GDPR)** | GDPR is a European Union regulation that protects the personal data of EU citizens and residents. |
| **Health Information Portability and Accountability Act (HIPAA)** | HIPAA assures that an individual’s health information is properly protected by setting use and disclosure standards. |
| **California Consumer Privacy Act (CCPA)** | The CCPA is a state statute intended to enhance privacy rights and consumer protection for residents of California, United States. The CCPA is the first state statute to require businesses to provide consumers with the ability to opt-out of the sale of their personal information. |

</div>

<!--s-->

## Ethics of Data Science | Examining Bias

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Why is bias a concern in data science?

- Bias can lead to unfair or discriminatory outcomes.
- Bias can perpetuate stereotypes and reinforce inequality.

### How can you identify and address bias in data science?

- Examine the data for bias in the collection or labeling process.
- [Fairness-aware machine learning](https://en.wikipedia.org/wiki/Fairness_(machine_learning)) to mitigate bias.

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>
<img src="https://storage.googleapis.com/gweb-uniblog-publish-prod/images/gender-before-after.width-1000.format-webp.webp" style="margin: 0 auto; display: block; width: 100%; border-radius: 5px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Google, 2018</p>
</div>
</div>
</div>

<!--s-->

## Ethics of Data Science | Accountability

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### **Scenario**: A self-driving car causes an accident, resulting in injury or death.

### Who should be held accountable?

- The manufacturer of the car.
- The developer of the software.
- The owner of the car.
- The government.

</div>
<div class="c2 col-centered" style = "width: 50%">
<div>
<img src="https://dda.ndus.edu/ddreview/wp-content/uploads/sites/18/2021/10/selfDriving.png" style="margin: 0 auto; display: block; width: 100%; border-radius: 5px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Nygard, 2021</p>
</div>


</div>
</div>

<!--s-->

<div class = "col-wrapper">
<div class="c1" style = "width: 70%; font-size: 1em;">

## Ethics of Data Science | Radicalization and Misinformation

### How can data science be used to **radicalize** and **misinform** people?

- By manipulating data to support false narratives.
- By targeting vulnerable populations with misleading information.
- By hyper-recommending content that reinforces extremist views.

### How can you combat radicalization and misinformation in data science?

- Fact-checking and verifying sources.
- Promoting trust, media literacy, and critical thinking.
- Implementing algorithms that prioritize accuracy and credibility.

</div>
<div class="c2 col-centered" style = "width: 30%">

<div>
<img src="https://p16-va-tiktok.ibyteimg.com/obj/musically-maliva-obj/4a2a1776a08f761c6464f596c0c5e8e6.png" style="margin: 0 auto; display: block; width: 50%; border-radius: 5px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">TikTok, 2024</p>
</div>
</div>
</div>

<!--s-->

## Ethics of Data Science | Recap

1. Consent for Data
2. Privacy
3. Examining Bias
4. Accountability
5. Radicalization and Misinformation

<!--s-->

<div class = "header-slide">

# Career Tips

</div>

<!--s-->

<div class = "header-slide">

## Effective Networking

</div>

<!--s-->

<div class = "header-slide">

## Monetizing Your Curiosity

</div>

<!--s-->

<div class = "header-slide">

## Building a Personal Brand

</div>

<!--s-->