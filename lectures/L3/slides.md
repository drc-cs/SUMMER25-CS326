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
  ## L3

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

# Data Preprocessing

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you in taking a raw dataset and transforming it into a machine learning-ready dataset?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

## Common Data Format

Often, your goal is to build a complete, numerical matrix with rows (instances) and columns (features).

- **Ideal**: Raw Dataset → Model → Task
- **Reality**: Raw Dataset → ML-Ready Dataset → Features (?) → Model → Task

Once you have a numerical matrix, you can apply modeling methods to it.

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.93 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.65 |
| ... | ... | ... | ... | ... |

<!--s-->

## Overview | From Raw Dataset to ML-Ready Dataset

The raw dataset is transformed into a machine learning-ready dataset by converting the data into a format that can be used by machine learning models. This typically involves converting the data into numerical format, removing missing values, and scaling the data.

Each row should represent an instance (e.g. a student) and each column should represent a consistently-typed and formatted feature (e.g. homework score, exam score). By convention, the final column often represents the target variable (e.g. final exam score).

<br><br><br>

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; font-size: 0.8em;">

### Raw Dataset:

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| "2549@nw.edu" | 90%| 89% | None | 93% |
| "7856@nw.edu" | 79%| 75% | 70% | 65% |
| ... | ... | ... | ... | ... |

</div>
<div class="c2" style = "width: 50%;  font-size: 0.8em;">

### ML-Ready:

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.93 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.65 |
| ... | ... | ... | ... | ... |

</div>
</div>

<!--s-->

## Overview | Adding Feature Engineering

Feature engineering is the process of using domain knowledge to select and transform raw data into features that increase the predictive power of the learning models.

You can often consider feature engineering as a way to add new columns to your dataset that may help the model better understand the data (i.e. spoon-feeding a model with your domain knowledge). <br><br><br>

<div class = "col-wrapper">
<div class="c1" style = "width: 50%;  font-size: 0.8em;">

### ML-Ready:

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.79 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.73 |
| ... | ... | ... | ... | ... |


</div>
<div class="c2" style = "width: 50%;  font-size: 0.8em;">

### ML-Ready + Engineered Feature:

| Student ID | Homework | Exam 1 | Exam 2 | Average Exam Score | Final |
|------------|----------|--------|--------|---------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.79 | 0.93 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.74 | 0.65 |
| ... | ... | ... | ... | ... | ... |

</div>
</div>

<!--s-->

## Overview | Interaction Terms with PolynomialFeatures

PolynomialFeatures generates a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form 

$$[a, b]$$

the degree-2 polynomial features are 

$$ [1, a, b, a^2, ab, b^2] $$

this is a very quick way to add interaction terms to your dataset.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(degree=2, interaction_only=False,  include_bias=True)
poly.fit_transform(X)
```

```
# Original
[[0 1]
 [2 3]
 [4 5]]
```

```
# Transformed
array([[ 1.,  0.,  1.,  0.,  0.,  1.],
       [ 1.,  2.,  3.,  4.,  6.,  9.],
       [ 1.,  4.,  5., 16., 20., 25.]])
```

<!--s-->

## Data Transformations

Today we will cover a number of transformations and feature engineering methods that are common in the data preprocessing stage.

<div style="font-size: 0.75em">

| Type | Transformation | Description |
|-----------|----------------|-------------|
| Numerical | Binarization | Convert numeric to binary. |
| Numerical | Binning | Group numeric values. |
| Numerical | Log Transformation | Manage data scale disparities. |
| Numerical | Scaling | Standardize or scale features. |
| Categorical | One-Hot Encoding | Convert categories to binary columns. |
| Categorical | Feature Hashing | Compress categories into hash vectors. |
| Temporal | Temporal Features | Convert to bins, manage time zones. |
| Missing | Imputation | Fill missing values. [L2](https://drc-cs.github.io/cs326/lectures/L02_data_sources/#/25).|
| High-Dimensional | Feature Selection | Choose relevant features. |
| High-Dimensional | Feature Sampling | Reduce feature space via feature sampling. |
| High-Dimensional | Random Projection | Reduce feature space via random projection. |
| High-Dimensional | Principal Component Analysis (PCA) | Reduce dimensions while preserving data variability. |
</div>

<p style="text-align: left; font-size: 0.8em;">* Text data will be covered during the NLP lectures</p>

<!--s-->

## Numerical Data | Binarization

Convert numerical values to binary values via a threshold. Values above the threshold are set to 1, below threshold are set to 0.

```python
from sklearn.preprocessing import Binarizer

transformer = Binarizer(threshold=3).fit(data)
transformer.transform(data)
```

<img src="https://storage.googleapis.com/cs326-bucket/lecture_6/binarized.png" width="800" style="display: block; margin: 0 auto; border-radius: 10px;">

<!--s-->

## Numerical Data | Binning

Group numerical values into bins.

- **Uniform**: Equal width bins. Use this when the data is uniformly distributed.
- **Quantile**: Equal frequency bins. Use this when the data is not uniformly distributed, or when you want to be robust to outliers.

```python
from sklearn.preprocessing import KBinsDiscretizer

# uniform binning
transformer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform').fit(data)
transformer.transform(data)

# quantile binning
transformer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile').fit(data)
transformer.transform(data)
```

<!--s-->

## Numerical Data | Binning

<img src="https://storage.googleapis.com/cs326-bucket/lecture_6/binning.png" style="display: block; margin: 0 auto; border-radius: 10px;">

<!--s-->

## Numerical Data | Log Transformation

Logarithmic transformation of numerical values. This is useful for data with long-tailed distributions, or when the scale of the data varies significantly.


```python
import numpy as np
transformed = np.log(data)
```

<br><br>

<img src="https://storage.googleapis.com/slide_assets/long-tail.png" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Komorowski, 2016</p>


<!--s-->

## Numerical Data | Scaling

Standardize or scale numerical features.

- **MinMax**: Squeeze values into [0, 1].

$$ x_{\text{scaled}} = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)} $$

- **Standard**: Standardize features to have zero mean and unit variance:

$$ x_{\text{scaled}} = \frac{x - \mu}{\sigma} $$

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# min-max scaling
scaler = MinMaxScaler().fit(data)
scaler.transform(data)

# standard scaling
scaler = StandardScaler().fit(data)
scaler.transform(data)
```

<!--s-->

## Numerical Data | Scaling

| Method | Pros | Cons | When to Use | 
|--------|------|------| ------------|
| MinMax | Bounded between 0 and 1. | Sensitive to outliers | When the data is uniformly distributed. |
| Standard | More robust to outliers. | Not bounded. | When the data is normally distributed. |

**A special note on training / testing sets** -- always fit the scaler on the training set and transform both the training and testing sets using those parameters. This ensures that the testing set is not used to influence the training set.

<!--s-->

## Numerical Data | Why Scale Data?

### Convergence
Some models (like SVMs and neural networks) converge faster when the data is scaled. Scaling the data can help the model find the optimal solution more quickly.

### Performance
Some models (like KNN) are sensitive to the scale of the data. Scaling the data can improve the performance of these models. Consider the KNN model -- if the data is not scaled, the model will give more weight to features with larger scales.

### Regularization
Regularization methods (like L1 and L2 regularization) penalize large coefficients. If the features are on different scales, the regularization term may penalize some features more than others.

<!--s-->

## Categorical Data | One-Hot Encoding

Convert categorical features to binary columns. This will create a binary column for each unique value in the data. For example, the color feature will be transformed into three columns: red, green, and blue.

```python
from sklearn.preprocessing import OneHotEncoder

data = [["red"], ["green"], ["blue"]]
encoder = OneHotEncoder().fit(data)
encoder.transform(data)
```


```
array([[0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.]])
```

<!--s-->

## Categorical Data | Feature Hashing

Compress categorical features into hash vectors -- in other words, this method reduces the dimensionality of the feature space. Hashing is using a function that maps categorical values to fixed-length vectors.

Compared to one-hot encoding, feature hashing is more memory-efficient and can handle high-dimensional data. It is typically used when some categories are unknown ahead of time, or when the number of categories is very large.

However, it can lead to collisions, where different categories are mapped to the same column. In this case below, we are hashing the color feature into a 2-dimensional vector. 


```python
from sklearn.feature_extraction import FeatureHasher

data = [{"color": "red"}, {"color": "green"}, {"color": "blue"}]
hasher = FeatureHasher(n_features=2, input_type='dict').fit(data)
hasher.transform(data)
```

```
array([[-1,  0],
       [ 0,  1],
       [ 1,  0]])
```

<!--s-->

## L3 | Q1

<div class='col-wrapper' style = 'display: flex; align-items: top;'>
<div class='c1' style = 'width: 60%; display: flex; flex-direction: column;'>

You are working with a dataset that contains a feature with 100 unique categories. You are unsure if all categories are present in the training set, and you want to reduce the dimensionality of the feature space. Which method would you use?

<div style = 'line-height: 2em;'>
&emsp;A. One-Hot Encoding <br>
&emsp;B. Feature Hashing <br>
</div>
</div>
<div class="c2" style="width: 50%; height: 100%;">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L3 | Q1" width="100%" height="100%" style="border-radius: 10px"></iframe>
</div>
</div>

<!--s-->

## Temporal Data | Temporal Granularity

Converting temporal features to bins based on the required granularity for your model helps manage your time series data. For example, you can convert a date feature to year / month / day, depending on the required resolution of your model.

We'll cover more of time series data handling in our time series analysis lecture -- but pandas has some great tools for this!

```python
import pandas as pd

data = pd.DataFrame({"date": ["2021-01-01", "2021-02-01", "2021-03-01"]})
data["date"] = pd.to_datetime(data["date"])
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day

```

date|month|day
---|---|---
2021-01-01|1|1
2021-02-01|2|1
2021-03-01|3|1

<!--s-->

## Missing Data | Imputation

We covered imputation in more detail in [L1](https://drc-cs.github.io/SUMMER25-CS326/lectures/L1/#).

<div style = "font-size: 0.85em;">


| Method | Description | When to Use |
| --- | --- | --- |
| Forward / backward fill | Fill missing value using the last / next valid value. | Time Series |
| Imputation by interpolation | Use interpolation to estimate missing values. | Time Series |
| Mean value imputation | Fill missing value with mean from column. | Random missing values |
| Conditional mean imputation | Estimate mean from other variables in the dataset. | Random missing values |
| Random imputation | Sample random values from a column. | Random missing values | 
| KNN imputation | Use K-nearest neighbors to fill missing values. | Random missing values |
| Multiple Imputation | Uses many regression models and other variables to fill missing values. | Random missing values |

</div>

<!--s-->

<div class="header-slide">

# Dimensionality Reduction

</div>

<!--s-->

## High-Dimensional Data | Why Reduce Dimensions?

- **Curse of Dimensionality**: As the number of features increases, the amount of data required to cover the feature space grows exponentially.

- **Overfitting**: High-dimensional data is more likely to overfit the model, leading to poor generalization.

- **Computational Complexity**: High-dimensional data requires more computational resources to process.

- **Interpretability**: High-dimensional data can be difficult to interpret and visualize.

<!--s-->

## High-Dimensional Data | The Curse of Dimensionality

**tldr;** As the number of features increases, the amount of data required to cover the feature space grows exponentially. This can lead to overfitting and poor generalization.

**Intuition**: Consider a 2D space with a unit square. If we divide the square into 10 equal parts along each axis, we get 100 smaller squares. If we divide it into 100 equal parts along each axis, we get 10,000 smaller squares. The number of smaller squares grows exponentially with the number of divisions. Without exponentially growing data points for these smaller squares, a model needs to make more and more inferences about the data.

**Takeaway**: With regards to machine learning, this means that as the number of features increases, the amount of data required to cover the feature space grows exponentially. Given that we need more data to cover the feature space effectively, and we rarely do, this can lead to overfitting and poor generalization.

<img src="https://storage.googleapis.com/slide_assets/dimensionality.png" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Rajput, 2012</p>

<!--s-->

## High-Dimensional Data | Feature Selection

One method to reduce dimensionality is to choose relevant features based on their importance for classification. The example below uses the chi-squared test to select the 20 best features. 

This works by selecting the features that are least likely to be independent of the class label (i.e. the features that are most likely to be relevant for classification).


```python
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2

X, y = load_digits(return_X_y=True)
X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
```

<!--s-->

## High-Dimensional Data | Feature Sampling

Another way is to reduce the feature space using modeling methods. The example below uses a random forest classifier to select the most important features. 

This works because the random forest model is selecting the features that are most likely to be important for classification. We'll cover random forests in more detail in a future lecture.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:1010/1*R3oJiyaQwyLUyLZL-scDpw.png" width="300" style="display: block; margin: 0 auto; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Deniz Gunay, 2023</p>

</div>
<div class="c2" style = "width: 70%">

```python
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

X, y = load_digits(return_X_y=True)
X_new = SelectFromModel(RandomForestClassifier()).fit_transform(X, y)
```

</div>
</div>


<!--s-->

## High-Dimensional Data | Random Projection

Gaussian random projection is a type of random projection that uses a Gaussian distribution to generate the random matrix. It takes the original data and projects it into a lower-dimensional space.

Johnson-Lindenstrauss Lemma states that a random projection can preserve the pairwise distances between the data points with high probability (up to a constant factor).

<img src = "https://storage.googleapis.com/slide_assets/gaussian_projection.png" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Ratra, 2022</p>

```python
from sklearn.random_projection import GaussianRandomProjection

transformer = GaussianRandomProjection(n_components=2).fit(data)
transformer.transform(data)
```

<!--s-->

## High-Dimensional Data | Principal Component Analysis (PCA)

PCA reduces dimensions while preserving data variability. PCA works by finding the principal components of the data, which are the directions in which the data varies the most. It then projects the data onto these principal components, reducing the dimensionality of the data while preserving as much of the variability as possible.

<img src = "https://numxl.com/wp-content/uploads/principal-component-analysis-pca-featured.png" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NumXL</p>

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(data)
pca.transform(data)
```

<!--s-->

## L3 | Q2


<div class = "col-wrapper">
<div class="c1" style = "width: 60%">

Here is an example where PCA was utilized to reduce the dimensionality of a dataset from 3 to 2 dimensions.

The data was projected onto the first two principal components, which captured the most variability in the data (80% in PCA1 and 15% in PCA2).

What is one issue you can think of with using PCA1 or PCA2 as features in a model? Hint: Think about the interpretability of the features.

<img src = "https://storage.googleapis.com/slide_assets/pca_example.png" width="800" style="display: block; margin: 0 auto; border-radius: 10px;">

</div>
<div class="c2 col-centered" style = "width: 40%">

<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L3 | Q2" width="100%" height="100%" style="border-radius: 10px"></iframe>

</div>
</div>


<!--s-->

## Summary

- 🔥 A common mistake is standardizing the entire dataset before splitting it into training and testing sets. Always fit the scaler on the training set and transform both the training and testing sets using those parameters.

- There are a number of ways that you can transform your data, regardless of type, to make it more suitable for machine learning models. Having more transformations in your toolbox can help you better understand your data and build better models.

- There is no one-size-fits-all solution to data preprocessing. The methods you choose will depend on the data you have and the model you are building.

<!--s-->

<div class="header-slide">

# Supervised Machine Learning I

</div>

<!--s-->

## Supervised Machine Learning

Supervised machine learning is a type of machine learning where the model is trained on a labeled dataset. Assume we have some features $X$ and a target variable $y$. The goal is to learn a mapping from $X$ to $y$.

$$ y = f(X) + \epsilon $$

Where $f(X)$ is the true relationship between the features and the target variable, and $\epsilon$ is a random error term. The goal of supervised machine learning is to estimate $f(X)$.

`$$ \widehat{y} = \widehat{f}(X) $$`

Where $\widehat{y}$ is the predicted value and $\widehat{f}(X)$ is the estimated function.

<!--s-->

## Supervised Machine Learning

To learn the mapping from $X$ to $y$, we need to define a model that can capture the relationship between the features and the target variable. The model is trained on a labeled dataset, where the features are the input and the target variable is the output.

- Splitting Data
  - Splitting data into training, validation, and testing sets.
- Linear Regression
  - Fundamental regression algorithm.
- Logistic Regression
  - Extension of linear regression for classification.
- k-Nearest Neighbors (KNN)
  - Fundamental classification algorithm.

<!--s-->

## Splitting Data into Training, Validation, and Testing Sets

Splitting data into training, validation, and testing sets is crucial for model evaluation and selection.

- **Training Set**: Used for fitting the model.
- **Validation Set**: Used for parameter tuning and model selection.
- **Test Set**: Used to evaluate the model performance.

A good, general rule of thumb is to split the data into 70% training, 15% validation, and 15% testing. In practice, k-fold cross-validation is often used to maximize the use of data. We will discuss this method in future lectures.

<img src="https://miro.medium.com/v2/resize:fit:1160/format:webp/1*OECM6SWmlhVzebmSuvMtBg.png" width="500" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Chavan, 2023</p>

<!--s-->

## Methods of Splitting

<div class = "col-wrapper">

<div class="c1" style = "width: 60%; font-size: 0.8em;">

### Random Split
Ensure you shuffle the data to avoid bias. Important when your dataset is ordered.

### Stratified Split
Used with imbalanced data to ensure each set reflects the overall distribution of the target variable. Important when your dataset has a class imbalance.

### Time-Based Split
Used for time series data to ensure the model is evaluated on future data. Important when your dataset is time-dependent.

### Group-Based Split
Used when data points are not independent, such as in medical studies. Important when your dataset has groups of related data points.

</div>

<div class="c2 col-centered" style = "width: 40%;">

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

<img src="https://miro.medium.com/v2/resize:fit:1160/format:webp/1*OECM6SWmlhVzebmSuvMtBg.png" width="500" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Chavan, 2023</p>

</div>

<!--s-->

## Why Split on Groups?

Imagine we're working with a medical dataset aimed at predicting the likelihood of a patient having a particular disease based on various features, such as age, blood pressure, cholesterol levels, etc.

### Dataset

- **Patient A:** 
  - Visit 1: Age 50, Blood Pressure 130/85, Cholesterol 210
  - Visit 2: Age 51, Blood Pressure 135/88, Cholesterol 215

- **Patient B:**
  - Visit 1: Age 60, Blood Pressure 140/90, Cholesterol 225
  - Visit 2: Age 61, Blood Pressure 145/92, Cholesterol 230

<!--s-->

## Incorrect Splitting

- **Training Set:**
  - Patient A, Visit 1
  - Patient B, Visit 1

- **Testing Set:**
  - Patient A, Visit 2
  - Patient B, Visit 2

In this splitting scenario, the model could learn specific patterns from Patient A and Patient B in the training set and then simply recall them in the testing set. Since it has already seen data from these patients, even with slightly different features, **it may perform well without actually generalizing to unseen patients**.

<!--s-->

## Correct Splitting

- **Training Set:**
  - Patient A, Visit 1
  - Patient A, Visit 2

- **Testing Set:**
  - Patient B, Visit 1
  - Patient B, Visit 2

In these cases, the model does not have prior exposure to the patients in the testing set, ensuring an unbiased evaluation of its performance. It will need to apply its learning to truly "new" data, similar to real-world scenarios where new patients must be diagnosed based on features the model has learned from different patients.

<!--s-->

## Key Takeaways for Group-Based Splitting

- **Independence:** Keeping data separate between training and testing sets maintains the independence necessary for unbiased model evaluation.

- **Generalization:** This approach ensures that the model can generalize its learning from one set of data to another, which is crucial for effective predictions.

<!--s-->

<div class="header-slide">

# Linear Regression

</div>

<!--s-->

## Linear Regression | Concept

Linear regression attempts to model the relationship between two or more variables by fitting a linear equation to observed data. The components to perform linear regression: 

$$ \widehat{y} = X\beta $$

Where $ \widehat{y} $ is the predicted value, $ X $ is the feature matrix, and $ \beta $ is the coefficient vector. The goal is to find the coefficients that minimize the error between the predicted value and the actual value.

<img src="http://www.stanford.edu/class/stats202/figs/Chapter3/3.1.png" width="600" style="margin: 0 auto; display: block;">

<!--s-->

## Linear Regression | Cost Function

The objective of linear regression is to minimize the cost function $ J(\beta) $:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

$$ J(\beta) = \frac{1}{2m} \sum_{i=1}^m (\widehat{y}_i - y_i)^2 $$

Where $ \widehat{y} = X\beta $ is the prediction. This is most easily solved by finding the normal equation solution:

$$ \beta = (X^T X)^{-1} X^T y $$

The normal equation is derived by setting the gradient of $J(\beta) $ to zero. This is a closed-form solution that can be computed directly.

</div>
<div class="c2" style = "width: 50%">

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

<img src="http://www.stanford.edu/class/stats202/figs/Chapter3/3.1.png" width="100%">

</div>
</div>

<!--s-->

## Linear Regression | Normal Equation Notes

### Adding a Bias Term

Practically, if we want to include a bias term in the model, we can add a column of ones to the feature matrix $ X $. Your H3 will illustrate this concept.

$$ \widehat{y} = X\beta $$

### Gradient Descent

For large datasets, the normal equation can be computationally expensive. Instead, we can use gradient descent to minimize the cost function iteratively. We'll talk about gradient descent within the context of logistic regression later today.

<!--s-->

## Linear Regression | Regression Model Evaluation

To evaluate a regression model, we can use metrics such as mean squared error (MSE), mean absolute error (MAE), mean absolute percentage error (MAPE), and R-squared.

<div style = "font-size: 0.8em;">

| Metric | Formula | Notes |
| --- | --- | --- |
| Mean Squared Error (MSE) | `$\frac{1}{m} \sum_{i=1}^m (\widehat{y}_i - y_i)^2$` | Punishes large errors more than small errors. |
| Mean Absolute Error (MAE) | `$\frac{1}{m} \sum_{i=1}^m \|\widehat{y}_i - y_i\|$` | Less sensitive to outliers than MSE. |
| Mean Absolute Percentage Error (MAPE) | `$\frac{1}{m} \sum_{i=1}^m \left \| \frac{\widehat{y}_i - y_i}{y_i} \right\| \times 100$` | Useful for comparing models with different scales. |
| R-squared | `$1 - \frac{\sum(\widehat{y}_i - y_i)^2}{\sum(\bar{y} - y_i)^2}$` | Proportion of the variance in the dependent variable that is predictable from the independent variables. |

</div>

<!--s-->

## Linear Regression | Pros and Cons

### Pros

- Simple and easy to understand.
- Fast to train.
- Provides a good, very interpretable baseline model.

### Cons

- Assumes a linear relationship between the features and the target variable.
- Sensitive to outliers.

<!--s-->

## Linear Regression | A Brief Note on Regularization

<div style = "font-size: 0.8em;">
Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function. The two most common types of regularization are L1 (Lasso) and L2 (Ridge) regularization.

Recall that the cost function for linear regression is:

$$ J(\beta) = \frac{1}{2m} \sum_{i=1}^m (\widehat{y}_i - y_i)^2 $$

**L1 Regularization**: Adds the absolute value of the coefficients to the cost function. This effectively performs feature selection by pushing some coefficients towards zero.

$$ J(\beta) = J(\beta) + \lambda \sum_{j=1}^n |\beta_j| $$

**L2 Regularization**: Adds the square of the coefficients to the cost function. This shrinks the coefficients, but does not set them to zero. This is useful when all features are assumed to be relevant.

$$ J(\beta) = J(\beta) + \lambda \sum_{j=1}^n \beta_j^2 $$
</div>

<!--s-->

## L3 | Q4

<div class = "col-wrapper">

<div class="c1" style = "width: 50%">

When is L2 regularization (Ridge) preferred over L1 regularization (Lasso)?

A. When all features are assumed to be relevant.<br>
B. When some features are assumed to be irrelevant.

</div>

<div class="c2" style = "width: 50%">

<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L7 | Q1" width="100%" height="100%" style="border-radius: 10px;"></iframe>

</div>

</div>

<!--s-->

<div class="header-slide">

# Logistic Regression

</div>

<!--s-->

## Logistic Regression | Concept

Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic function.

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*G3imr4PVeU1SPSsZLW9ghA.png" width="400" style="margin: 0 auto; display: block;">
<span style="font-size: 0.8em; text-align: center; display: block; color: grey;">Joshi, 2019</span>

<!--s-->

## Logistic Regression | Formula

This model is based on the sigmoid function $\sigma(z)$:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Where 

$$ z = X\beta $$

Note that $\sigma(z)$ is the probability that the dependent variable is 1 given the input $X$. Consider the similar form of the linear regression model:

$$ \widehat{y} = X\beta $$

The key difference is that the output of logistic regression is passed through the sigmoid function to obtain a value between 0 and 1, which can be interpreted as a probability. This works because the sigmoid function maps any real number to the range [0, 1]. While linear regression predicts the value of the dependent variable, logistic regression predicts the probability that the dependent variable is 1. 

<!--s-->

## Logistic Regression | No Closed-Form Solution

In linear regression, we can calculate the optimal coefficients $\beta$ directly. However, in logistic regression, we cannot do this because the sigmoid function is non-linear. This means that there is no closed-form solution for logistic regression.

Instead, we use gradient descent to minimize the cost function. Gradient descent is an optimization algorithm that iteratively updates the parameters to minimize the cost function, and forms the basis of many machine learning algorithms.

<img src="https://machinelearningspace.com/wp-content/uploads/2023/01/Gradient-Descent-Top2-1024x645.png" width="500" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">machinelearningspace.com (2013)</p>

<!--s-->

## Logistic Regression | Cost Function

The cost function used in logistic regression is the cross-entropy loss:

$$ J(\beta) = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\widehat{y}_i) + (1 - y_i) \log(1 - \widehat{y}_i)] $$

$$ \widehat{y} = \sigma(X\beta) $$

Let's make sure we understand the intuition behind the cost function $J(\beta)$.

If the true label ($y$) is 1, we want the predicted probability ($\widehat{y}$) to be close to 1. If the true label ($y$) is 0, we want the predicted probability ($\widehat{y}$) to be close to 0. The cost goes up as the predicted probability diverges from the true label.

<!--s-->

## Logistic Regression | Gradient Descent

To minimize $ J(\beta) $, we update $ \beta $ iteratively using the gradient of $ J(\beta) $:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; font-size: 0.8em;">

$$ \beta := \beta - \alpha \frac{\partial J}{\partial \beta} $$

Where $ \alpha $ is the learning rate, and the gradient $ \frac{\partial J}{\partial \beta} $ is:

$$ \frac{\partial J}{\partial \beta} = \frac{1}{m} X^T (\sigma(X\beta) - y) $$

Where $ \sigma(X\beta) $ is the predicted probability, $ y $ is the true label, $ X $ is the feature matrix, $ m $ is the number of instances, $ \beta $ is the coefficient vector, and $ \alpha $ is the learning rate.

This is a simple concept that forms the basis of many gradient-based optimization algorithms, and is widely used in deep learning. 

Similar to linear regression -- if we want to include a bias term, we can add a column of ones to the feature matrix $ X $.

</div>
<div class="c2 col-centered" style = "width: 50%">

<img src="https://machinelearningspace.com/wp-content/uploads/2023/01/Gradient-Descent-Top2-1024x645.png" width="500" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">machinelearningspace.com (2013)</p>

</div>
</div>

<!--s-->

## L3 | Q5

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Okay, so let's walk through an example. Suppose you have already done the following: 

1. Obtained the current prediction ($\widehat{y}$) with $ \sigma(X\beta) $.
2. Calculated the gradient $ \frac{\partial J}{\partial \beta} $.

What do you do next?


</div>
<div class="c2" style = "width: 50%">

<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L7 | Q2" width="100%" height="100%" style="border-radius: 10px;"></iframe>

</div>
</div>

<!--s-->

## Logistic Regression | Classifier

Once we have the optimal coefficients, we can use the logistic function to predict the probability that the dependent variable is 1. 

We can then use a threshold to classify the instance as 0 or 1 (usually 0.5). The following code snippet shows how to use the scikit-learn library to fit a logistic regression model and make predictions.


```python
from sklearn.linear_model import LogisticRegression

X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 0, 1, 1]
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)
logistic_regression_model.predict([[3.5, 3.5]])
```

```python
array([[0.3361201, 0.6638799]])
```

<!--s-->

## L3 | Q6

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

Our logistic regression model was trained with $X = [[1, 2], [2, 3], [3, 4], [4, 5]]$ and $y = [0, 0, 1, 1]$. We then made a prediction for the point $[3.5, 3.5]$.

What does this output represent?

```python
array([[0.3361201, 0.6638799]])
```

</div>
<div class="c2" style = "width: 50%">
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L7 | Q3" width="100%" height="100%" style="border-radius: 10px;"></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# K-Nearest Neighbors (KNN)

</div>

<!--s-->

## k-Nearest Neighbors (KNN) | Concept

KNN is a non-parametric method used for classification (and regression!).

The principle behind nearest neighbor methods is to find a predefined number of samples closest in distance to the new point, and predict the label from these using majority vote.

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_7/knn_setup.html"></iframe>

<!--s-->

## k-Nearest Neighbors (KNN) | What is it doing?

Given a new instance $ x' $, KNN classification computes the distance between $ x' $ and all other examples. The k closest points are selected and the predicted label is determined by majority vote.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; font-size: 0.7em;">

### Euclidean Distance

`$ d(x, x') =\sqrt{\sum_{i=1}^n (x_i - x'_i)^2} $`

### Manhattan Distance

`$ d(x, x') = \sum_{i=1}^n |x_i - x'_i| $`

### Cosine Distance 

`$ d(x, x') = 1 - \frac{x \cdot x'}{||x|| \cdot ||x'||} $`

### Jaccard Distance (useful for categorical data!)

`$ d(x, x') = 1 - \frac{|x \cap x'|}{|x \cup x'|} $`

### Hamming Distance (useful for strings!)

`$ d(x, x') = \frac{1}{n} \sum_{i=1}^n x_i \neq x'_i $`

</div>

<div class="c2 col-centered" style = "width: 60%;">

```python
from sklearn.neighbors import KNeighborsClassifier

# Default is Minkowski distance.
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
```
</div>
</div>

<!--s-->

## k-Nearest Neighbors (KNN) | Example

Given the following data points (X) and their corresponding labels (y), what is the predicted label for the point (3.5, 3.5) using KNN with k=3?

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 50%">

<p style="margin-left: -10em;">A. 0<br><br>B. 1</p>

</div>
<div class="c2" style = "width: 50%">

<iframe width = "100%" height = "50%" src="https://storage.googleapis.com/cs326-bucket/lecture_7/knn_actual.html" padding=2em;></iframe>

```python
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 0, 1, 1]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
knn.predict([[3.5, 3.5]])
```

</div>
</div>

<!--s-->

## k-Nearest Neighbors (KNN) | Hyperparameters

### Number of Neighbors (k)

The number of neighbors to consider when making predictions.

### Distance Metric
The metric used to calculate the distance between points.

### Weights
Uniform weights give equal weight to all neighbors, while distance weights give more weight to closer neighbors.

### Algorithm
The algorithm used to compute the nearest neighbors. Some examples include Ball Tree, KD Tree, and Brute Force.

<!--s-->

## k-Nearest Neighbors (KNN) | Pros and Cons

### Pros

- Simple and easy to understand.
- No training phase.
- Can be used for both classification and regression.

### Cons

- Computationally expensive.
- Sensitive to the scale of the data.
- Requires a large amount of memory.

<!--s-->

## k-Nearest Neighbors (KNN) | Classification Model Evaluation

To evaluate a binary classification model like this, we can use metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.

| Metric | Formula | Notes |
| --- | --- | --- | 
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Easy to interpret but flawed.
| Precision | $\frac{TP}{TP + FP}$ | Useful when the cost of false positives is high. |
| Recall | $\frac{TP}{TP + FN}$ | Useful when the cost of false negatives is high. |
| F1 Score | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Harmonic mean of precision and recall. | 
| ROC-AUC | Area under the ROC curve. | Useful for imbalanced datasets. |

<!--s-->

## Summary

- We discussed the importance of splitting data into training, validation, and test sets.
- We delved into k-Nearest Neighbors, Linear Regression, and Logistic Regression with Gradient Descent, exploring practical implementations and theoretical foundations.
- Understanding these foundational concepts is crucial for advanced machine learning and model fine-tuning! 

<!--s-->