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
  ## L2

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

<div class = "header-slide">

# Correlation

</div>

<!--s-->

## Relationships in Data

Our goal is often to understand the relationships between different variables in our data.

- **Correlation**: Quantitatively measures the degree to which two variables move in relation to each other.

- **Association**: Describes the relationship between two variables in terms of co-occurrence.

<!--s-->

## Correlation

Correlation is a fundamental concept in data science. It is used to measure the strength and direction of a relationship between two variables.

<div style="text-align: center; height: 60%">
  <img src="https://www.scribbr.com/wp-content/uploads/2022/07/Perfect-positive-correlation-Perfect-negative-correlation.webp">
</div>

<!--s-->

## Correlation | Quantitative Measurement via Covariance

**Covariance** is a measure of how much two random variables vary together, which is a measure of their **correlation**.

The covariance between two variables \(X\) and \(Y\) can be defined as:

<div class="col-centered">
$ \text{cov}(X, Y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n} $
</div>

And it's really just a generalization of the variance to two variables:

<div class="col-centered">

$ \sigma^2 = \frac{\sum_{i=1}^n (X_i - \mu)^2}{n} $

</div>

<!--s-->

## Correlation | Interpreting Covariance

When the covariance is positive, it means that the two variables are moving in the same direction. When the covariance is negative, it means that the two variables are moving in opposite directions.

**But** size of the covariance is not standardized, so it is difficult to interpret the strength of the relationship. Consider the following example:

**Case 1:**
- **Study Hours (X):** <span class="code-span">[5, 10, 15, 20, 25]</span>
- **Test Scores (Y):** <span class="code-span">[50, 60, 70, 80, 90]</span>

**Case 2:**
- **Study Hours (X):** <span class="code-span">[5, 10, 15, 20, 25]</span>
- **Test Scores (Y):** <span class="code-span">[500, 600, 700, 800, 900]</span>

Covariance will be different in these cases, but the relationship is the same!

<!--s-->

## Correlation | Pearson Correlation Coefficient

Pearson correlation coefficient, denoted by \(r\), is a measure of the linear correlation between two variables. It ranges from -1 to 1, and so it is a **standardized** measure of the strength of the relationship.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2}\sqrt{\sum_i (y_i - \bar{y})^2}} $

<span class="code-span">r = 1</span>: Perfect positive linear relationship <br>
<span class="code-span">r = -1</span>: Perfect negative linear relationship <br>
<span class="code-span">r = 0</span>: No linear relationship



</div>
<div class="c2" style = "width: 50%">

<img src="https://www.scribbr.com/wp-content/uploads/2022/07/Perfect-positive-correlation-Perfect-negative-correlation.webp">

</div>
</div>

<!--s-->

## Correlation | Pearson Correlation Coefficient

Pearson's correlation coefficient is a great method to measure the strength of a linear relationship between two variables. However, it has some limitations:

- Sensitive to outliers
- It only measures linear relationships
- It is not robust to non-normality

If your data is not normally distributed, your relationship is not linear, or you have big outliers, you may want to consider another correlation method (e.g., Spearman's rank correlation coefficient).

<!--s-->

## Correlation | Spearman Rank Correlation Coefficient

Spearman Rank Correlation Coefficient counts the number of disordered pairs, not how well the data fits a line. Thus, it is better for non-linear relationships. You can use the formula below only if all n ranks are distinct integers.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<div>
$ r_s = 1 - \frac{6 \sum_i d_i^2}{n^3 - n} $
</div>
<div>
$ d_i = \text{rank}(x_i) - \text{rank}(y_i) $
</div>


<span class="code-span">r_s = 1</span>: Perfect positive relationship <br>
<span class="code-span">r_s = -1</span>: Perfect negative relationship <br>
<span class="code-span">r_s = 0</span>: No relationship

</div>
<div class="c2" style = "width: 50%">

<img src="https://www.scribbr.com/wp-content/uploads/2021/08/monotonic-relationships.png">

</div>
</div>


<!--s-->

## Correlation | Spearman Rank Correlation Coefficient

Spearman Rank Correlation Coefficient counts the number of disordered pairs, not how well the data fits a line. Thus, it is better for non-linear relationships. You can use the formula below only if all n ranks are distinct integers.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<div>
$ r_s = 1 - \frac{6 \sum_i d_i^2}{n^3 - n} $
</div>
<div>
$ d_i = \text{rank}(x_i) - \text{rank}(y_i) $
</div>


<span class="code-span">r_s = 1</span>: Perfect positive relationship <br>
<span class="code-span">r_s = -1</span>: Perfect negative relationship <br>
<span class="code-span">r_s = 0</span>: No relationship

</div>
<div class="c2" style = "width: 50%">

<img src="https://datatab.net/assets/tutorial/spearman/Calculate_Spearman_rank_correlation.png">
<p style="text-align: center; font-size: 0.6em; color: grey;"> Source: Datatab</p>

</div>
</div>


<!--s-->

## Correlation | Coefficients vs. Regression

- Correlation coefficients measure the strength of a linear relationship between two variables.
- Regression analysis can be used to describe the relationship between a dependent variable and one or more independent variables.

<!--s-->

## L4 | Q2

What is a benefit of using Pearson's correlation coefficient over covariance?

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 50%; padding-bottom: 20%;">

<div style = "line-height: 2em; font-size: 0.8em;">
&emsp;A. It is a standardized measure of the strength of the relationship. <br>
&emsp;B. It is not sensitive to outliers. <br>
&emsp;C. It is not affected by the scale of the variables. <br>
&emsp;D. It is not affected by the distribution of the data. <br>
</div>

</div>

<div class="c2 col-centered" style = "width: 50%;">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L4 | Q2" width = "100%" height = "100%"></iframe>

</div>
</div>

<!--s-->

<div class="header-slide">

# Simpson's Paradox

</div>

<!--s-->

## Simpson's Paradox

Simpson's Paradox is a phenomenon in probability and statistics, in which a trend appears in different groups of data but **disappears** or **reverses** when these groups are combined.

Recall that correlation measures the strength of a linear relationship between two variables. But, always remember that correlation does not imply causation! 

**Simpson's Paradox** is a situation where a relationship is reversed when the data is split into subgroups.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*8tP_5zRKNAyVSeexu7RJZg.png">

<!--s-->

## Simpson's Paradox | Recovery Rate Example

Which hospital would you rather have surgery in? A or B?

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 16 | 784 | 2% |
| B | 63 | 2037 | 3% |

<!--s-->

## Simpson's Paradox | Recovery Rate Example

If you are in good condition, which hospital would you rather have surgery in? A or B?

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 592 | 1.3% |
| B | 6 | 594 | 1% |

<!--s-->

## Simpson's Paradox | Recovery Rate Example

If you are in poor condition, which hospital would you rather have surgery in? A or B?

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 192 | 4% |
| B | 57 | 1443 | 3.8% |

<!--s-->

## Simpson's Paradox | Recovery Rate Example

Let's look at all of the data together. Hospital B has a higher death rate than Hospital A in aggregate. But, when we look at the subgroups, Hospital A has a higher death rate in both subgroups.

<div style="font-size: 0.8em;">

### Overall

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 16 | 784 | 2% |
| B | 63 | 2037 | **3%** |

### Good Condition

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 592 | **1.3%** |
| B | 6 | 594 | 1% |

### Poor Condition

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 192 | **4%** |
| B | 57 | 1443 | 3.8% |

</div>

<!--s-->

## Simpson's Paradox | Linear Regression Example

Simpson's Paradox can also occur in linear regression or correlation analysis.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*8tP_5zRKNAyVSeexu7RJZg.png">

<!--s-->

## Simpson's Paradox | What Is (Typically) Happening?

1. **Confounding Variables**: The relationship between the variables is influenced by a third variable.
2. **Sample Size**: The sample size of the subgroups is not large enough to capture the true relationship.

<!--s-->

## Simpson's Paradox | Prevention

1. **Segment Data Carefully**: Understand the context and how data groups are formed.
2. **Identify Confounders**: Look for variables that might be influencing the results.
3. **Holistic Approach**: Consider both combined and segmented data analyses.
4. **Use Visualizations**: Visualizations can help identify patterns and trends.

<!--s-->

<div class="header-slide">

# Association Analysis

</div>

<!--s-->

## Association Analysis | Definition

Association analysis measures the strength of co-occurrence between one item and another. It is widely applied in retail analysis of transactions, recommendation engines, online clickstream analysis, and more.

<!--s-->

## Association Analysis | Explanation

Given a set of transactions, association analysis finds rules that will predict the occurrence of an item based on the occurrences of other items.

For example, if a customer buys a product, what other products are they likely to buy?

<!--s-->

## Association Analysis | Definitions

- **Itemset**: A collection of one or more items
  - Example: <span class="code-span">{Milk, Bread, Diaper}</span>
  - **k-itemset**: An itemset that contains <span class="code-span">k</span> items
- **Association Rule**: An implication expression of the form <span class="code-span">X --> Y</span>, where <span class="code-span">X</span> and <span class="code-span">Y</span> are itemsets.
  - Example: <span class="code-span">{Milk, Diaper} --> {Beer}</span>

<!--s-->

## Association Rule | Evaluation

For a rule $ X \rightarrow Y $, where $ X \cap Y = \emptyset $, we can evaluate the rule using the following measures:

- **Support (S)**: Fraction of transactions that contain both X and Y. Where $\(T\)$ is the total number of transactions and $\sigma(X, Y)$ is the number of transactions that contain both $\(X\)$ and $\(Y\)$.

<div class="col-centered" style = "padding: 0.5em;">
$ S(X \rightarrow Y) = \frac{\sigma(X, Y)}{|T|} $
</div>

- **Confidence (C)**: Measures how often items in Y appear in transactions that contain X.

<div class="col-centered" style = "padding: 0.5em;">
$ C(X \rightarrow Y) = \frac{\text{S}(X, Y)}{\text{S}(X)} $
</div>

- **Lift (L)**: Takes into account the frequency of Y besides the confidence.

<div class="col-centered" style = "padding: 0.5em;">
$L(X \rightarrow Y) = \frac{S(X, Y)}{S(X)S(Y)}$
</div>

<!--s-->

## Association Rule | Example

Consider the following transactions:

| TID | Items |
| --- | --- |
| 1 | Bread, Milk |
| 2 | Bread, Diaper, Beer, Eggs |
| 3 | Milk, Diaper, Beer, Coke |
| 4 | Bread, Milk, Diaper, Beer |
| 5 | Bread, Milk, Diaper, Coke |

And the following association rule: <span class="code-span">{Milk, Diaper} --> {Beer}</span>

$ S = \frac{\sigma{\text(Milk, Diaper, Beer)}}{|T|} = \frac{2}{5} = 0.4 $

$ C = \frac{S(Milk, Diaper, Beer)}{S(Milk, Diaper)} = \frac{0.4}{0.6} = 0.67$

$ L = \frac{S(Milk, Diaper, Beer)}{S(Milk, Diaper)S(Beer)} = \frac{0.4}{0.6*0.6} = 1.11 $


<!--s-->

## Association Analysis Rule Generation

Given a set of transactions \(T\), the goal of association rule mining is to find all rules having:

- Support $ \geq $ Support threshold
- Confidence $ \geq $ Confidence threshold

The goal is to find all rules that satisfy these constraints. Lift is often used as a measure of the *interestingness* of the rule. Aka how much more likely is Y given X than if Y were independent of X.

<!--s-->

## Association Analysis Rule Generation | Brute-force Approach

In order to get all of the possible association rules, we would need to:

  - List all possible association rules
  - Compute the support and confidence for each rule
  - Prune rules that fail the support or confidence thresholds

But, as with many ideal or simple solutions, this is computationally prohibitive.

<!--s-->

## Association Analysis Rule Generation | Apriori Principle

The Apriori principle is a fundamental concept in association rule mining. It states that if an itemset is frequent, then all of its subsets must also be frequent. 

This principle allows us to reduce the number of itemsets we need to consider when generating association rules, and thus reduce the computational complexity. Modern software implementations will use the Apriori principle to generate association rules.

<div class="col-centered">
<img src="https://storage.googleapis.com/slide_assets/apriori.png" style="border-radius:15px; height: 20%; width: 50%;">
</div>
<p style="text-align: center; font-size: 0.6em; color: grey;">Tank, Darshan. (2014)</p>

<!--s-->

## L4 | Q2

What is otherwise defined as the *interestingness* of an association rule?

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 50%; padding-bottom: 20%;">

<div style = "line-height: 2em; font-size: 0.8em;">
&emsp;A. Confidence <br>
&emsp;B. Support <br>
&emsp;C. Lift<br>
</div>

</div>

<div class="c2 col-centered" style = "width: 50%;">

<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L4 | Q2" width = "100%" height = "100%"></iframe>

</div>
</div>

<!--s-->

## Summary

- **Correlation** measures the strength of a relationship between two variables.
- **Simpson's Paradox** is a phenomenon where a trend appears in different groups of data but disappears or reverses when these groups are combined.
- **Association Analysis** measures the strength of co-occurrence between one item and another.

<!--s-->

<!--s-->

<div class="header-slide">

# Hypothesis Testing

</div>

<!--s-->

## What is Hypothesis Testing?

Hypothesis testing uses statistical methods to determine whether there is enough evidence to reject a null hypothesis in favor of an alternative hypothesis.

The null hypothesis ($H_0$) is a statement that there is no difference between groups, while the alternative hypothesis ($H_1$) posits that there is a difference.

<!--s-->

## Question 1.1.0 | Hypothesis Testing

**Question:** You have data on the average time it takes for two groups of students to complete a test. Group A has an average time of 45 minutes +/- 5 minutes, while Group B has an average time of 50 minutes +/- 5 minutes. You want to determine if there is a significant difference between the two groups. 

<div class='col-wrapper' style = "max-height: 50vh;">
<div class='c1' style = 'width: 50%;'>

What test would you use to compare the two groups, assuming all conditions are met?

&emsp;A. Chi-Square Test of Independence<br>
&emsp;B. One-Way ANOVA<br>
&emsp;**C. T-Test (Independent)**<br>
&emsp;D. T-Test (Paired)<br>
</div>
<div class='c2 col-centered' style = 'width: 50%;'>
<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L05 | Q1.1.0" height = "100%"></iframe>
</div>
</div>

<!--s-->

## Question 1.1.1 | Hypothesis Testing

**Question:** You have data on the average time it takes for two groups of students to complete a test. Group A has an average time of 45 minutes +/- 5 minutes, while Group B has an average time of 50 minutes +/- 5 minutes. You want to determine if there is a significant difference between the two groups. 

<div class='col-wrapper'  style = "max-height: 50vh;">
<div class='c1' style = 'width: 50%;'>

What percentage confidence would you give your answer to Question 1.1.0?
</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L05 | Q1.1.1" height = "100%"></iframe>
</div>
</div>

<!--s-->

## Common Hypothesis Tests

The following are some hypothesis tests used in statistics:

<div style="font-size: 0.7em;">

| Test | Assumptions | Usage (Easy ~ Rule) |
| --- | --- | --- |
| t-test | 1. Data are independently and identically distributed. <br> 2. Both groups follow a normal distribution. <br> 3. Variances across groups are approximately equal.* | When comparing the means of two independent groups. |
| t-test (paired)  | 1. Data are independently and identically distributed. <br> 2. The differences are normally distributed.<br> 3. The pairs are selected randomly and are representative.| When you have pre / post test information on subjects or a matched pairs experiment. |
| chi-square test of independence | 1. Data are independently and identically distributed. <br> 2. All empirical frequencies are 5 or greater. | When comparing proportions across categories. |
| One-way ANOVA  | 1. Responses for each group are normally distributed. <br> 2. Variances across groups are approximately equal. <br> 3. Data are independently and identically distributed. | When comparing the means of three or more groups. |

</div>

<!--s-->

## Question 1.2.0 | Hypothesis Testing

**Question:** You have data on the average time it takes for two groups of students to complete a test. Group A has an average time of 45 minutes +/- 5 minutes, while Group B has an average time of 50 minutes +/- 5 minutes. You want to determine if there is a significant difference between the two groups. 

<div class='col-wrapper' style = "max-height: 50vh;">
<div class='c1' style = 'width: 50%;'>

What test would you use to compare the two groups, assuming all conditions are met?

&emsp;A. Chi-Square Test of Independence<br>
&emsp;B. One-Way ANOVA<br>
&emsp;C. T-Test (Independent)<br>
&emsp;D. T-Test (Paired)<br>
</div>
<div class='c2 col-centered' style = 'width: 50%;'>
<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L05 | Q1.2.0" height = "100%"></iframe>
</div>
</div>

<!--s-->

## Question 1.2.1 | Hypothesis Testing

**Question:** You have data on the average time it takes for two groups of students to complete a test. Group A has an average time of 45 minutes +/- 5 minutes, while Group B has an average time of 50 minutes +/- 5 minutes. You want to determine if there is a significant difference between the two groups. 


<div class='col-wrapper'  style = "max-height: 50vh;">
<div class='c1' style = 'width: 50%;'>

What percentage confidence would you give your answer to Question 1.2.0?
</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L05 | Q1.2.1" height = "100%"></iframe>
</div>
</div>

<!--s-->


## Question 2 | Hypothesis Testing

**Question:** Suppose I wanted to do an analysis on the efficacy of Slide 10. Let's say I collected your level of confidence of Question 1 before and after seeing Slide 10. 

<div class='col-wrapper'>
<div class='c1' style = 'width: 50%;'>

What test would you use to compare the two groups, assuming all conditions are met?

&emsp;A. Chi-Square Test of Independence<br>
&emsp;B. One-Way ANOVA<br>
&emsp;C. T-Test (Independent)<br>
&emsp;D. T-Test (Paired)<br>
</div>
<div class='c2' style = 'width: 50%;'>
<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L05 | Q2" height="100%" ></iframe>
</div>
</div>

<!--s-->

## Common Hypothesis Tests | T-Test Setup

<div style="font-size: 0.9em">

### Scenario

Comparing the effect of two medications. Medication A has been used on 40 subjects, having an average recovery time of 8 days, with a standard deviation of 2 days. Medication B (new) has been used on 50 subjects, with an average recovery time of 7 days and a standard deviation of 2.5 days. 

### Hypotheses

- H0: μ1 = μ2 (No difference in mean recovery time)
- H1: μ1 ≠ μ2 (Difference in mean recovery time)

### Assumptions

- Groups are I.I.D.
    - I.I.D. stands for independent and identically distributed.
- Both groups follow a normal distribution.*
    - Once you have enough samples, the central limit theorem will ensure normality.
- Equal variances between the two groups (homoscedasticity).*
    - If variances are not equal, a Welch's t-test can be used.

</div>

<!--s-->

## Common Hypothesis Tests | T-Test Calculation

<div style="font-size: 0.9em">

### T-Statistic (Equal Variances)

`$$ t = \frac{\bar{x}_1 - \bar{x}_2}{s_p \sqrt{\frac{1}{n_1} + \frac{1}{n_2}}} $$`

`$$ s_p = \sqrt{\frac{(n_1 - 1)s_1^2 + (n_2 - 1)s_2^2}{n_1 + n_2 - 2}} $$`

Where:

- $\bar{x}_1$ and $\bar{x}_2$ are the sample means.
- $n_1$ and $n_2$ are the sample sizes.
- $s_p$ is the pooled standard deviation.
- $s_1$ and $s_2$ are the sample standard deviations. <br>

### Degrees of Freedom (Equal Variances)

The degrees of freedom for this t-test is calculated as: 

$$ df = n_1 + n_2 - 2 $$

</div>

<!--s-->

## Common Hypothesis Tests | T-Test Decision

### Decision Process

1. Compare the computed t-value against the critical t-value from the t-distribution table with $\alpha = 0.05$ and $df$.
2. If the computed t-value is above the critical t-value, reject the null hypothesis.

<div class="col-centered">
<img src="https://storage.googleapis.com/slide_assets/ttest_decision.png" style="border-radius: 10px; height: 50%; width: 50%;">
</div>

<!--s-->

## Common Hypothesis Tests | T-Test (Paired) Setup

### Scenario

A group of 25 patients is measured for cholesterol levels before and after a particular treatment, aiming to evaluate the treatment's effect on cholesterol.

### Hypotheses

- H0: $d=0$ (No difference in mean cholesterol levels)
- H1: $d \ne 0$ (Difference in mean cholesterol levels)

### Assumptions

- The differences within pairs are independent.
- The differences are normally distributed.
- The pairs are selected randomly and are representative.

<!--s-->

## Common Hypothesis Tests | T-Test (Paired) Calculation

### Paired T-Statistic

First, find the difference ($d$) for each pair. Then, calculate the mean ($\bar{d}$) and standard deviation ($s_d$) of those differences.

$$ t = \frac{\bar{d}}{s_d / \sqrt{n}} $$

where $n$ is the number of pairs.

### Degrees of Freedom

Degrees of freedom can be calculated with $df = n - 1$.

<!--s-->

## Common Hypothesis Tests | T-Test (Paired) Decision

### Decision Process

1. Using the t-distribution table with $df = n - 1$, compare the calculated t-value.
2. If the computed t-value falls within the critical range, reject the null hypothesis.

<div class="col-centered">
<img src="https://storage.googleapis.com/slide_assets/ttest_decision.png" style="border-radius: 10px; height: 50%; width: 50%;">
</div>


<!--s-->

## Common Hypothesis Tests | Chi-Square Test Setup

### Scenario

You have two penguin species, Adelie and Chinstrap. They are observed in the wild, and the following data is collected from two islands (A and B):

| Species | Island A | Island B |
|---------|----------|----------|
| Adelie  | 15       | 5       |
| Chinstrap | 5     | 15       |


### Hypotheses

- H0: The species distribution is independent of the island.
- H1: The species distribution is dependent on the island.

### Assumptions

- Observations are independent.
- All expected frequencies are at least 5.

<!--s-->

## Question 3 | Chi-Square Expectation Calculation

**Question:** Calculate the expected frequency for Adelie penguins on Island A. Assuming the null hypothesis is true, what is the expected frequency?

<div class='col-wrapper' style = "max-height: 50vh;">
<div class='c1' style = 'width: 50%;'>

| Species | Island A | Island B |
|---------|----------|----------|
| Adelie  | 15       | 5       |
| Chinstrap | 5     | 15       |

&emsp;A. 10<br>
&emsp;B. 5<br>
&emsp;C. 7.5<br>
&emsp;D. 12.5<br>
</div>
<div class='c2 col-centered' style = 'width: 50%;'>
<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L05 | Q3" height = "100%"></iframe>
</div>
</div>

<!--s-->

## Common Hypothesis Tests | Chi-Square Test Calculation

### Chi-Square Statistic

The chi-square statistic of independence is calculated as:

$$\chi^2 = \sum \frac{(O - E)^2}{E}$$

Where:
- $O$ is the observed frequency.
- $E$ is the expected frequency, which is calculated as the row total times the column total divided by the grand total.

### Degrees of Freedom

$$df = (r - 1) \times (c - 1)$$

Where:
- $r$ is the number of rows.
- $c$ is the number of columns.

<!--s-->

## Common Hypothesis Tests | Chi-Square Test Calculation

### Calculation ($\chi^2$)

$$ \chi^2 = \frac{(15 - 10)^2}{10} + \frac{(5 - 10)^2}{10} + \frac{(5 - 10)^2}{10} + \frac{(15 - 10)^2}{10} = 10 $$

### Degrees of Freedom ($df$)

$$ df = (2 - 1) \times (2 - 1) = 1 $$

<!--s-->

## Common Hypothesis Tests | Chi-Square Test Decision

### Decision Process

1. Compare the $\chi^2$ value against the critical values from the chi-square distribution table with $df$
2. If $\chi^2 > \chi_{critical}$, reject H0.

<div class="col-centered">
<img src="https://www.mun.ca/biology/scarr/IntroPopGen-Table-D-01-smc.jpg" style = "border-radius: 10px; height: 40%; width: 40%;">
</div>
<p style="text-align: center; font-size: 0.6em; color: grey;">© 2022, Steven M. Carr</p>

<!--s-->

## Common Hypothesis Tests | One-Way ANOVA Setup

### Scenario

Testing if there's a difference in the mean test scores across three teaching methods used across different groups.

### Hypotheses


- H0: $ \mu_1 = \mu_2 = \mu_3 $ (no difference among the group means)
- H1: At least one group mean is different.

### Assumptions

- Groups are I.I.D.
- Groups follow a normal distribution.
- Variances across groups are approximately equal.
    - A good rule of thumb is a ratio of the largest to the smallest variance less than 4.

<!--s-->

## Common Hypothesis Tests | One-Way ANOVA Calculation

### F-Statistic

Anova breaks down the variance explained by the groups ($SS_{between}$) and the variance not explained by the groups ($SS_{within}$). The F-statistic measures the ratio of the variance between groups to the variance within groups:

$$ F = \frac{SS_{between} / df_{between}}{SS_{within} / df_{within}} $$

The total sum of squares (SS) is calculated as:

$$ s^2 = \frac{SS}{df} = \frac{\sum (x - \bar{x})^2}{n - 1} $$

Where:
- $SS$ is the sum of squares.
- $df$ is the degrees of freedom.
- $x$ is the data point.
- $\bar{x}$ is the sample mean.
- $n$ is the sample size.

<!--s-->

## Common Hypothesis Tests | One-Way ANOVA Calculation

### Degrees of Freedom

The degrees of freedom are $df_{between} = k - 1$ and $df_{within} = N - k$.

Where:
- $k$ is the number of groups.
- $N$ is the total number of observations.


<!--s-->

## Common Hypothesis Tests | One-Way ANOVA Decision

### Decision Process

1. Compare the calculated F-value with the critical F-value from the F-distribution table at $df_{between}$ and $df_{within}$.
2. Reject H0 if $F > F_{critical}$, indicating significant differences among means.

<div class="col-centered">
<img src="https://i0.wp.com/statisticsbyjim.com/wp-content/uploads/2022/02/F-table_Alpha05.png?resize=817%2C744&ssl=1" height = "50%" width = "50%" style="border-radius: 10px;">
</div>

<!--s-->

## Choosing a Non-Parametric Test

If the assumptions for parametric tests are not met, non-parametric tests can be used. 

These tests are distribution-free and do not require the data to be normally distributed. These may make less powerful inferences than parametric tests, because parametric tests derive power from the strong assumptions they make about the shape of the data.

<div style="font-size: 0.8em">

| Test    | Use in place of | Description |
|-----------------------|------------------|-------------------------|
| Spearman’s r  | Pearson’s r | For quantitative variables with non-linear relation. |
| Kruskal–Wallis H  | ANOVA | For 3 or more groups of quantitative data |
| Mann-Whitney U | Independent t-test  | For 2 groups, different populations. |
| Wilcoxon Signed-rank  | Paired t-test| For 2 groups from the same population. |

<p style = "text-align: center; color: grey"> © Adapted from Scribbr, 2024 </p>

</div>

<!--s-->

<div class="header-slide">

# A/B Testing

</div>

<!--s-->

## Introduction to A/B Testing

A/B testing, a critical methodology widely used in tech companies and various industries, plays a pivotal role in optimizing user experience and boosting key performance indicators such as engagement and conversion rates.

This approach tests two variants, A and B, in a randomized experiment to make data-driven decisions regarding changes to products, websites, or apps.

<div class="col-centered">
<img src = "https://blog.christianposta.com/images/abtesting.png" style="border-radius:15px; height: 20%; width: 50%;">
</div>
<p style="text-align: center; font-size: 0.6em; color: grey;">Posta (2024)</p>

<!--s-->

## Understanding A/B Testing

At its core, A/B testing involves comparing two versions of a web page, product feature, or other elements to determine which one performs better.

This process enables businesses to make incremental changes that can lead to significant improvements in user satisfaction and business outcomes.

<div class="col-centered">
<img src = "https://blog.christianposta.com/images/abtesting.png" style="border-radius:15px; height: 20%; width: 50%;">
</div>
<p style="text-align: center; font-size: 0.6em; color: grey;">Posta (2024)</p>

<!--s-->

## Why Implement A/B Testing?

- **Innovation and Improvement:** Whether it's deciding which photograph leads to better retention of subscribers or testing a new feature's impact on user engagement, A/B testing provides a systematic method for assessing the potential benefits of new ideas.

- **Data-driven Decisions:** By relying on empirical data, companies can move beyond guesswork and make informed decisions that align with their strategic goals.

<!--s-->



## Case Study | Social Network App

A study within a new social network app revealed that users who tagged a friend during their trial period had a **31**% increase in daily usage after the trial period ended.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

This insight suggests that encouraging social interactions may **increase user retention**.

An immediate reaction might be to prioritize features solely based on this correlation, such as pushing all new installs to tag a friend.

</div>
<div class="c2" style = "width: 50%">

<iframe width = "100%" height = "60%" src="https://lottie.host/embed/6a06ca79-cb9c-48e2-bdac-a8bbbcb03815/j0VNBrf81N.json"></iframe>

</div>
</div>

<!--s-->

## Question 4 | A/B Testing Considerations

After observing the correlation between tagging a friend and increased user retention, should the app developers immediately implement this feature for all new users?

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 60%; padding-bottom: 20%;">

<div style = "line-height: 2em;">
&emsp;A. Yes <br>
&emsp;B. No <br>
</div>
</div>

<div class="c2 col-centered" style = "width: 40%; padding-bottom: 20%">
<iframe src = "https://drc-cs-9a3f6.firebaseapp.com/?label=L05 | Q4" width = "100%" height = "100%"></iframe>
</div>
</div>

<!--s-->

## Case Study | Considering External Influences

Factors such as:

- **Seasonality:** Changes in user behavior due to holidays or seasonal trends.
- **Marketing Campaigns:** The impact of marketing campaigns on user engagement.
- **Product Updates:** Changes in the app's features or design.
- **User Demographics:** Variations in user behavior based on age, location, or other factors.

Could all contribute to conversion rate changes. Thus, it's imperative not to jump to conclusions about causality based solely on passively observed correlations.

<!--s-->

## Case Study | Implementing Effective A/B Testing

To understand the impact of tagging a friend during a trial period, a structured A/B testing approach is essential:

1. **Construct a Hypothesis:** Begin with a clear, testable hypothesis.
    - **H0**: Tagging a friend during the trial period has no effect on the original users' retention.
    - **H1**: Tagging a friend during the trial period increases the original users' retention.

2. **Design Study / Trial:** Establish how many subjects are necessary for a statistically significant test and estimate the duration of the experiment. 
    - Resources exist to make this process easier [[link]](https://www.evanmiller.org/ab-testing/sample-size.html)

3. **Measure Results:** Use hypothesis testing to assess if there's a significant difference between the two groups.

4. **Take Action:** Based on the findings, make informed decisions and continue the cycle of experimentation and learning.

<!--s-->

## Proper A/B Testing Highlights

1. **Randomization:** Randomly assign subjects to the control and treatment groups to avoid selection bias.

2. **Statistical Significance:** Use statistical tests to determine if the observed differences are significant.

3. **Sample Size:** Ensure that the sample size is large enough to detect meaningful differences.

4. **Duration:** Run the experiment for a sufficient duration to capture the effects of the changes.

5. **Segmentation:** Analyze results across different segments to identify potential variations in the treatment effect.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 1em; left: 0; width: 50%; position: absolute;">

  # H2 | basic_statistics.py
  ## Code & Free Response

```bash
cd <class directory>
git pull
```

</div>
</div>
<div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 0%">
<iframe src="https://lottie.host/embed/6c677caa-d54a-411c-b0c0-6f186378d571/UKVVhf0EJN.json" height = "100%" width = "100%"></iframe>

</div>
</div>


<!--s-->

<div class="header-slide">

# Additional Slides
## (for the curious)

</div>

<!--s-->

## Wondering what a T Distribution Looks Like?

Here's how you can visualize a t-distribution in Python (and better understand T-tables!):

<div class = "col-wrapper">
<div class="c1" style = "width: 70%">

```python
import numpy as np
import plotly.express as px
from scipy.stats import t

x = np.linspace(-5, 5, 1000)
y = t.pdf(x, df=4)
fig = px.line(x=x, y=y)
fig.update_layout(title='t-distribution with 4 degrees of freedom', template = "plotly_white")
```

<img src="https://storage.googleapis.com/slide_assets/p-value.png">

</div>
<div class="c2" style = "width: 30%; font-size: 0.6em;">

PDF of the t-distribution can be calculated as:

$$ f(x, \nu) = \frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\Gamma\left(\frac{\nu}{2}\right)}\left(1+\frac{x^2}{\nu}\right)^{-\frac{\nu+1}{2}} $$

Where $\nu$ is the degrees of freedom.

</div>
</div>

<!--s-->

## Why df = n - 1?

We often subtract 1 from the sample size to account for the fact that we are estimating the population mean from the sample mean. But *why* do we do this?

To illustrate:

- Recall that the definition of degrees of freedom is the number of **independent** observations in a sample.
- Suppose you have a sample of 5 data points: <span class="code-span"> [1, 2, 3, 4, 5] </span>
- You calculate the sample mean: <span class="code-span"> 3 </span>
- You can calculate the last data point from the sample mean and 4 of the other data points. So you lose a degree of freedom, because you can't change the last data point without changing the sample mean.

<!--s-->

## Common Hypothesis Tests | T-Test Calculation

<div style="font-size: 0.8em">

### Calculating the T-Statistic (Not Equal Variances, Welch's T-Test)
  
`$$ t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}}} $$`
  
Where:  
  - $\bar{x}_1$ and $\bar{x}_2$ are the sample means.
  - $n_1$ and $n_2$ are the sample sizes.
  - $s_1$ and $s_2$ are the sample standard deviations. <br>

### Calculating the Degrees of Freedom (Not Equal Variances, Welch's T-Test)

$$ df = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{\left(\frac{s_1^2}{n_1}\right)^2}{n_1 - 1} + \frac{\left(\frac{s_2^2}{n_2}\right)^2}{n_2 - 1}} $$

Where:
  - $s_1$ and $s_2$ are the sample standard deviations.
  - $n_1$ and $n_2$ are the sample sizes.

</div>

<!--s-->