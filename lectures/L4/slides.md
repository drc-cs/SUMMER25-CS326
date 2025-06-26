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
  ## L4

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

## L4 | Supervised Machine Learning II

Today we're going to discuss: 

1. ROC-AUC
2. K-Fold Cross-Validation
3. Multi-Class & Multi-Label Classification

<!--s-->

<div class="header-slide">

# ROC-AUC

</div>

<!--s-->

## ROC-AUC

- **Receiver Operating Characteristic (ROC) Curve**: A graphical representation of the performance of a binary classifier system as its discrimination threshold is varied.

- **Area Under the Curve (AUC)**: The area under the ROC curve, which quantifies the classifier’s ability to distinguish between classes.

<img src="https://assets-global.website-files.com/6266b596eef18c1931f938f9/64760779d5dc484958a3f917_classification_metrics_017-min.png" width="400" style="margin: 0 auto; display: block;border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Source: Evidently AI</p>

<!--s-->

## ROC-AUC | Key Concepts

<div style="font-size: 0.7em;">

**True Positive Rate (TPR)**: The proportion of actual positive cases that are correctly identified by the classifier.<br>
**False Positive Rate (FPR)**: The proportion of actual negative cases that are incorrectly identified as positive by the classifier.<br>
**ROC**: When the FPR is plotted against the TPR for each binary classification threshold, we obtain the ROC curve.
</div>
<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*CQ-1ceyX80EE0a_s3SwvgQ.png" width="100%" style="margin: 0 auto; display: block;border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Cortex, 2020</p>

<!--s-->

## ROC-AUC | Key Concepts

<img src="https://miro.medium.com/v2/resize:fit:4512/format:webp/1*zNtuQziwUKkGUxhG0Go5kA.png" width="100%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Cortex, 2020</p>

<!--s-->

## L4 | Q1

I have a binary classifier that predicts whether or not a credit card transaction is fraud. The ROC-AUC of the classifier is 0.4. What does this mean?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
  &emsp;A. The classifier is worse than random.<br>
  &emsp;B. The classifier is random. <br>
  &emsp;C. The classifier is better than random. <br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L4 | Q1" width="100%" height="100%" style="border-radius: 10px;"></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# K-Fold Cross-Validation

</div>

<!--s-->

## K-Fold Cross-Validation

K-Fold Cross-Validation is a technique used to evaluate the performance of a machine learning model. It involves splitting the data into K equal-sized folds, training the model on K-1 folds, and evaluating the model on the remaining fold. This process is repeated K times, with each fold serving as the validation set once.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*AAwIlHM8TpAVe4l2FihNUQ.png" width="100%" style="margin: 0 auto; display: block;border-radius: 10px;">
<span style="font-size: 0.8em; text-align: center; display: block; color: grey;">Patro 2021</span>

<!--s-->

## K-Fold Cross-Validation | Advantages

When implemented correctly, K-Fold Cross-Validation has several advantages:

- **Better Use of Data**: K-Fold Cross-Validation uses all the data for training and validation, which can lead to more accurate estimates of model performance.

- **Reduced Variance**: By averaging the results of K different validation sets, K-Fold Cross-Validation can reduce the variance of the model evaluation.

- **Model Selection**: K-Fold Cross-Validation can be used to select the best model hyperparameters.

<!--s-->

<div class="header-slide">

# Multi-Class & Multi-Label Classification

</div>

<!--s-->

## Multi-Class Classification vs Multi-Label Classification

- **Multi-Class Classification**: A classification task with more than two classes, where each sample is assigned to one class.

- **Multi-Label Classification**: A classification task where each sample can be assigned to multiple classes.

<img src ="https://s46486.pcdn.co/wp-content/uploads/2022/06/1-3.jpg" width="80%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Thaker 2022</p>

<!--s-->

## Multi-Class Classification | Logistic Regression

In multi-class classification, each sample is assigned to one class. The goal is to predict the class label of a given sample. We've already seen at least one model that can be used for multi-class classification: Logistic Regression!

Logistic regression can be extended to multi-class classification using the softmax function.

<img src="https://storage.googleapis.com/slide_assets/multiclass.png" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Emerson</p>

<!--s-->

## Multi-Class Classification | Updating Logistic Regression

To use Logistic Regression for multi-class classification, we need to update the model. Recall that the logistic function is defined as:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

For multi-class classification, we use the softmax function instead:

$$ \sigma(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}} $$

Where:

- $z_j$ is the input to the $j$-th output unit.
- $K$ is the number of classes.

Intuitively, the softmax function converts the output of the model into probabilities for each class, with a sum of 1. An output may look like: 

<div style="text-align: center;">
<span class="code-span"> [0.1, 0.6, 0.3] </span>
</div>

<!--s-->

## Multi-Class Classification | Updating Logistic Regression

Recall binary cross-entropy:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} y^{(i)} \log(\widehat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \widehat{y}^{(i)}) $$

Alternatively, the cost function for multi-class classification is the cross-entropy loss:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\widehat{y}_k^{(i)}) $$


Where:

- $m$ is the number of samples.
- $K$ is the number of classes.
- $y_k^{(i)}$ is the true label of the $i$-th sample for class $k$.
- $\widehat{y}_k^{(i)}$ is the predicted probability of the $i$-th sample for class $k$.

<!--s-->

## Multi-Class Classification | Updating Logistic Regression

| Binary Classification | Multi-Class Classification |
|-----------------------|-----------------------------|
| Sigmoid Function      | Softmax Function            |
| Binary Cross-Entropy  | Cross-Entropy Loss          |
| One output unit, one class. The negative class is just 1 - positive class. | $K$ output units, $K$ classes  with probabilities that sum to 1. |
| One weight vector | $K$ weight vectors |

You don't need to implement the softmax function or cross-entropy loss from scratch for this course, but it's important to understand the intuition behind them!

<!--s-->

## Multi-Class Classification | Other Approaches

There are other approaches to multi-class classification, such as:

- **One-vs-All (OvA)**: Train $K$ binary classifiers, one for each class. During prediction, choose the class with the highest probability.

- **One-vs-One (OvO)**: Train $K(K-1)/2$ binary classifiers, one for each pair of classes. During prediction, choose the class with the most votes.

<img src="https://dezyre.gumlet.io/images/blog/multi-class-classification-python-example/image_504965436171642418833831.png?w=1100&dpr=2.0" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">ProjectPro</p>

<!--s-->

## Multi-Class Classification | One-vs-All

In One-vs-All (OvA), we train $K$ binary classifiers, one for each class. During prediction, we choose the class with the highest probability.

For example, if we have three classes (A, B, C), we would train three binary classifiers:

- Classifier 1: A vs. (B, C)
- Classifier 2: B vs. (A, C)
- Classifier 3: C vs. (A, B)

<img src="https://dezyre.gumlet.io/images/blog/multi-class-classification-python-example/image_504965436171642418833831.png?w=1100&dpr=2.0" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">ProjectPro</p>

<!--s-->

## Multi-Class Classification | One-vs-One

In One-vs-One (OvO), we train $K(K-1)/2$ binary classifiers, one for each pair of classes. During prediction, we choose the class with the most votes.

For example, if we have three classes (A, B, C), we would train three binary classifiers:

- Classifier 1: A vs. B
- Classifier 2: A vs. C
- Classifier 3: B vs. C

<img src="https://dezyre.gumlet.io/images/blog/multi-class-classification-python-example/image_504965436171642418833831.png?w=1100&dpr=2.0" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">ProjectPro</p>

<!--s-->

## Multi-Label Classification

In multi-label classification, each sample can be assigned to multiple classes. Some examples include:

- Tagging images with multiple labels.
- Predicting the genre of a movie.
- Classifying text into multiple categories.

<!--s-->

## Multi-Label Classification | Approaches

There are several approaches to multi-label classification:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Binary Relevance
Train a separate binary classifier for each label -- e.g. one classifier for each genre of a movie.

### Classifier Chains
Train a chain of binary classifiers, where each classifier predicts the next label -- e.g. predict the next genre of a movie based on the previous genre.

</div>
<div class="c2" style = "width: 50%">

### Label Powerset
Treat each unique label combination as a single class -- e.g. treat the combination of genres as a single class.


</div>
</div>

<!--s-->

## Multi-Label Classification | Comparisons

| Approach | Pros | Cons |
|----------|------|------|
| Binary Relevance | Simple, easy to implement | Ignores label dependencies |
| Classifier Chains | Considers label dependencies | Sensitive to label order |
| Label Powerset | Considers label dependencies | Exponential increase in classes |

<!--s-->

## L4 | Q2

Let's say we have a dataset of movie genres. We want to predict the genre(s) of a movie. Should we use multi-class or multi-label classification?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. Multi-Class <br>
&emsp;B. Multi-Label<br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L4 | Q2" width="100%" height="100%" style="border-radius: 10px;"></iframe>
</div>
</div>

<!--s-->

## L4 | Q3

Let's say we have a dataset of movie genres. We know that a movie can have only one genre, but there are many to choose from. Should we use multi-class or multi-label classification?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. Multi-Class<br>
&emsp;B. Multi-Label <br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L4 | Q3" width="100%" height="100%" style="border-radius: 10px;"></iframe>
</div>
</div>

<!--s-->

## Key takeaways:

### ROC-AUC
A key metric for evaluating binary classifiers.
### K-Fold Cross-Validation
A technique for evaluating machine learning models, and makes better use of the data.
### Multi-Class Classification
Involves predicting the class label of a given sample when there are more than two classes.
### Multi-Label Classification
Involves predicting multiple labels for a given sample.

<!--s-->

<div class="header-slide">

# L4 | Supervised Machine Learning III

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with the following methods:

  1. Decision Trees
  2. Ensemble Models

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class="header-slide">

# Decision Trees

</div>

<!--s-->

## Decision Trees | Overview

**Decision Trees** are a non-parametric supervised learning method used for classification and regression tasks. They are simple to **understand** and **interpret**, making them a popular choice in machine learning.

A decision tree is a tree-like structure where each internal node represents a feature or attribute, each branch represents a decision rule, and each leaf node represents the outcome of the decision.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220831135057/CARTClassificationAndRegressionTree.jpg" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks, 2024</p>

<!--s-->

## Decision Trees | ID3 Algorithm

The construction of a decision tree involves selecting the best feature to split the dataset at each node. One of the simplest algorithms for constructing categorical decision trees is the ID3 algorithm.

**ID3 Algorithm (Categorical Data)**:

1. Calculate the entropy of the target variable.
2. For each feature, calculate the information gained by splitting the data.
3. Select the feature with the highest information gain as the new node.

The ID3 algorithm recursively builds the decision tree by selecting the best feature to split the data at each node.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220831135057/CARTClassificationAndRegressionTree.jpg" height="40%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks, 2024</p>

<!--s-->

## Decision Trees | Entropy

**Entropy** is a measure of the impurity or disorder in a dataset. The entropy of a dataset is 0 when all instances belong to the same class and is maximized when the instances are evenly distributed across all classes. Entropy is defined as: 

$$ H(S) = - \sum_{i=1}^{n} p_i \log_2(p_i) $$

Where:

- $H(S)$ is the entropy of the dataset $S$.
- $p_i$ is the proportion of instances in class $i$ in the dataset.
- $n$ is the number of classes in the dataset.

<img src="https://miro.medium.com/v2/resize:fit:565/1*M15RZMSk8nGEyOnD8haF-A.png" width="40%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Provost, Foster; Fawcett, Tom. </p>

<!--s-->

## Decision Trees | Information Gain

**Information Gain** is a measure of the reduction in entropy achieved by splitting the dataset on a particular feature. IG is the difference between the entropy of the parent dataset and the weighted sum of the entropies of the child datasets.

$$ IG(S, A) = H(S) - H(S|A)$$

Where:

- $IG(S, A)$ is the information gain of splitting the dataset $S$ on feature $A$.
- $H(S)$ is the entropy of the parent dataset.
- $H(S | A)$ is the weighted sum of the entropies of the child datasets.


<br><br>
<img src="https://miro.medium.com/max/954/0*EfweHd4gB5j6tbsS.png" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">KDNuggets</p>


<!--s-->

## Decision Trees | ID3 Pseudo-Code

```text
ID3 Algorithm:
1. If all instances in the dataset belong to the same class, return a leaf node with that class.
2. If the dataset is empty, return a leaf node with the most common class in the parent dataset.
3. Calculate the entropy of the dataset.
4. For each feature, calculate the information gain by splitting the dataset.
5. Select the feature with the highest information gain as the new node.
6. Recursively apply the ID3 algorithm to the child datasets.
```

Please note, ID3 works for **categorical** data. For **continuous** data, we can use the C4.5 algorithm, which is an extension of ID3 that supports continuous data.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20220831135057/CARTClassificationAndRegressionTree.jpg" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks, 2024</p>

<!--s-->

## Decision Trees | How to Split Continuous Data

For the curious, here is a simple approach to split continuous data with a decision tree. ID3 won't do this out of the box, but C4.5 does have an implementation for it.

1. Sort the dataset by the feature value.
2. Calculate the information gain for each split point.
3. Select the split point with the highest information gain as the new node.

```text
Given the following continuous feature:

    [0.7, 0.3, 0.4]

First you sort it: 

    [0.3, 0.4, 0.7]

Then you evaluate information gain for your target variable at every split:

    [0.3 | 0.4 , 0.7]

    [0.3, 0.4 | 0.7]
```

<!--s-->

## Decision Trees | Overfitting

**Overfitting** is a common issue with decision trees, where the model captures noise in the training data rather than the underlying patterns. Overfitting can lead to poor **generalization** performance on unseen data.

<img src="https://gregorygundersen.com/image/linoverfit/spline.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Gunderson 2020</p>


<!--s-->

## Decision Trees | Overfitting

**Overfitting** is a common issue with decision trees, where the model captures noise in the training data rather than the underlying patterns. Overfitting can lead to poor **generalization** performance on unseen data.

**Strategies to Prevent Overfitting in Decision Trees**:

1. **Pruning**: Remove branches that do not improve the model's performance on the validation data.
    - This is similar to L1 regularization in linear models!
2. **Minimum Samples per Leaf**: Set a minimum number of samples required to split a node.
3. **Maximum Depth**: Limit the depth of the decision tree.
4. **Maximum Features**: Limit the number of features considered for splitting.

<!--s-->


## Decision Trees | Pruning

**Pruning** is a technique used to reduce the size of a decision tree by removing branches that do not improve the model's performance on the validation data. Pruning helps prevent overfitting and improves the generalization performance of the model.

Practically, this is often done by growing the tree to its maximum size and then remove branches that do not improve the model's performance on the validation data. A loss function is used to evaluate the performance of the model on the validation data, and branches that do not improve the loss are pruned: 

$$ L(T) = \sum_{t=1}^{T} L(y_t, \widehat{y}_t) + \alpha |T| $$

Where:

- $L(T)$ is the loss function of the decision tree $T$.
- $L(y_t, \widehat{y}_t)$ is the loss function of the prediction $y_t$.
- $\alpha$ is the regularization parameter.
- $|T|$ is the number of nodes in the decision tree.


<!--s-->

## Decision Tree Algorithm Comparisons

| Algorithm | Data Types | Splitting Criteria |
|-----------|------------|--------------------|
| ID3       | Categorical | Entropy & Information Gain    |
| C4.5      | Categorical & Continuous | Entropy & Information Gain |
| CART      | Categorical & Continuous | Gini Impurity |


<!--s-->

## L4 | Q4

Which of the following decision tree algorithms is a reasonable choice for continuous data?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. ID3 <br>
&emsp;B. C4.5 **<br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L4 | Q4" width="100%" height="100%" style="border-radius: 10px"></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# Ensemble Models

</div>

<!--s-->

## Ensemble Models | Overview

**Ensemble Models** combine multiple individual models to improve predictive performance. The key idea behind ensemble models is that a group of weak learners can come together to form a strong learner. Ensemble models are widely used in practice due to their ability to reduce overfitting and improve generalization.

We will discuss three types of ensemble models:

1. **Bagging**: Bootstrap Aggregating
2. **Boosting**: Sequential Training
3. **Stacking**: Meta-Learning

<!--s-->

## Ensemble Models | Bagging

**Bagging (Bootstrap Aggregating)** is an ensemble method that involves training multiple models on different subsets of the training data and aggregating their predictions. The key idea behind bagging is to reduce variance by averaging the predictions of multiple models.

**Intuition**: By training multiple models on different subsets of the data, bagging reduces the impact of outliers and noise in the training data. The final prediction is obtained by averaging the predictions of all models.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20230731175958/Bagging-classifier.png" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## Ensemble Models | Bagging | Example: Random Forests

**Random Forests** are an ensemble learning method that combines multiple decision trees to create a more robust and accurate model. Random Forests use bagging to train multiple decision trees on different subsets of the data and aggregate their predictions.

**Key Features of Random Forests**:
- Each decision tree is trained on a random subset of the features.
- The final prediction is obtained by averaging the predictions of all decision trees.

Random Forests are widely used in practice due to their robustness, scalability, and ability to handle high-dimensional data.

<img src="https://tikz.net/janosh/random-forest.png" height="40%" style="margin: 0 auto; display: block; padding: 10">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## Ensemble Models | Boosting

**Boosting** is an ensemble method that involves training multiple models sequentially, where each model learns from the errors of its predecessor.

**Intuition**: By focusing on the misclassified instances in each iteration, boosting aims to improve the overall performance of the model. The final prediction is obtained by combining the predictions of all models.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20210707140911/Boosting.png" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## Ensemble Models | Boosting | Example: AdaBoost

**AdaBoost (Adaptive Boosting)** is a popular boosting algorithm. AdaBoost works by assigning weights to each instance in the training data and adjusting the weights based on the performance of the model.

**Key Features of AdaBoost**:

1. Train a weak learner on the training data.
2. Increase the weights of misclassified instances.
3. Train the next weak learner on the updated weights.
4. Repeat the process until a stopping criterion is met.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20210707140911/Boosting.png" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## Ensemble Models | Boosting | Example: Gradient Boosting

**Gradient Boosting** is another popular boosting algorithm. Gradient Boosting works by fitting a sequence of weak learners to the residuals of the previous model. This differs from AdaBoost, which focuses on the misclassified instances. A popular implementation of Gradient Boosting is **XGBoost** (❤️).

**Key Features of Gradient Boosting**:

1. Fit a weak learner to the training data.
2. Compute the residuals of the model.
3. Fit the next weak learner to the residuals.
4. Repeat the process until a stopping criterion is met.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200721214745/gradientboosting.PNG" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## Ensemble Models | Stacking

**Stacking (Stacked Generalization)** is an ensemble method that combines multiple models using a meta-learner. The key idea behind stacking is to train multiple base models on the training data and use their predictions as input to a meta-learner.

**Intuition**: By combining the predictions of multiple models, stacking aims to improve the overall performance of the model. The final prediction is obtained by training a meta-learner on the predictions of the base models.

<img src="https://miro.medium.com/v2/resize:fit:946/1*T-JHq4AK3dyRNi7gpn9-Xw.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Setunga, 2020</p

<!--s-->

## Ensemble Models | Stacking | Example

**Stacked Generalization** involves training multiple base models on the training data and using their predictions as input to a meta-learner. The meta-learner is trained on the predictions of the base models to make the final prediction.

**Key Features of Stacked Generalization**:

1. Train multiple base models on the training data.
2. Use the predictions of the base models as input to the meta-learner.
3. Train the meta-learner on the predictions of the base models.
4. Use the meta-learner to make the final prediction.

Stacking is often trained end-to-end, where the base models and meta-learner are trained simultaneously to optimize the overall performance.

<img src="https://miro.medium.com/v2/resize:fit:946/1*T-JHq4AK3dyRNi7gpn9-Xw.png" height="30%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Setunga, 2020</p

<!--s-->

## L4 | Q5

Let's say that you train a sequence of models that learn from the mistakes of the predecessors. Instead of focusing on the misclassified instances (and weighting them more highly), you focus on improving the residuals. What algorithm is this most similar to?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
A. Bagging (e.g. Random Forest) <br>
B. Boosting (e.g. Adaboost) <br>
C. Boosting (e.g. Gradient Boosting)<br>
D. Stacking
</div>
</div>
<div class='c2' style = 'width: 40%;'>
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L4 | Q5" width="100%" height="100%" style="border-radius: 10px"></iframe>
</div>
</div>
<!--s-->

## Summary

In this lecture, we explored two fundamental concepts in supervised machine learning:

1. **Decision Trees**:
    - A non-parametric supervised learning method used for classification and regression tasks.
    - Can be constructed using the ID3 algorithm based on entropy and information gain.
    - Prone to overfitting, which can be mitigated using pruning and other strategies.

2. **Ensemble Models**:
    - Combine multiple individual models to improve predictive performance.
    - Include bagging, boosting, and stacking as popular ensemble methods.
    - Reduce overfitting and improve generalization by combining multiple models.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 1em; left: 0; width: 55%; position: absolute;">

  ## H3 | machine_learning.py
  ### Code & Free Response

```bash
cd <class directory>
git pull
```

</div>
</div>
<div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 0%; margin: 10px;">
<iframe src="https://lottie.host/embed/6c677caa-d54a-411c-b0c0-6f186378d571/UKVVhf0EJN.json" height = "100%" width = "100%"></iframe>

</div>
</div>

<!--s-->