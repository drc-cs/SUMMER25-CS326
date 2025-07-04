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
  ## L6

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

# Clustering

</div>

<!--s-->

## Agenda

Supervised machine learning is about predicting $y$ given $X$. Unsupervised machine learning is about finding patterns in $X$, without being given $y$.

There are a number of techniques that we can consider to be unsupervised machine learning, spanning a broad range of methods (e.g. clustering, dimensionality reduction, anomaly detection). Today we will focus on clustering.

### Clustering
1. Partitional Clustering
2. Hierarchical Clustering
3. Density-Based Clustering

<!--s-->

## Clustering | Applications

Clustering is a fundamental technique in unsupervised machine learning, and has a wide range of applications.

<div class = "col-wrapper">
<div class="c1" style = "width: 70%; font-size: 0.75em;">

<div>

| Application | Example | 
| --- | --- |
| Customer segmentation | Segmenting customers based on purchase history. |
| Document clustering | Grouping similar documents together. |
| Image segmentation | Segmenting an image into regions of interest. |
| Anomaly detection | Detecting fraudulent transactions. |
| Recommendation systems | Recommending products based on user behavior. |
</div>

</div>
<div class="c2" style = "width: 30%">

<div>
<img src="https://cambridge-intelligence.com/wp-content/uploads/2021/01/graph-clustering-800px.png" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Cambridge Intelligence, 2016</p>
</div>

</div>
</div>

<!--s-->

## Clustering | Goals

Clustering is the task of grouping a set of objects in such a way that objects in the same group (or cluster) are more similar to each other than to those in other groups.

A good clustering result has the following properties:

- High intra-cluster similarity
- Low inter-cluster similarity

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vEsw12wO0KxvYne0m4Cr5w.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Yehoshua, 2023</p>

<!--s-->

## Clustering | Revisiting Distance Metrics

> "Everything in data science seems to come down to distances and separating stuff." - Anonymous PhD

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; font-size: 0.75em;">

### Euclidean Distance

`$ d(x, x') =\sqrt{\sum_{i=1}^n (x_i - x'_i)^2} $`

### Manhattan Distance

`$ d(x, x') = \sum_{i=1}^n |x_i - x'_i| $`

### Cosine Distance 

`$ d(x, x') = 1 - \frac{x \cdot x'}{||x|| \cdot ||x'||} $`

</div>

<div class="c2" style = "width: 50%; font-size: 0.75em;">

### Jaccard Distance (useful for categorical data!)

`$ d(x, x') = 1 - \frac{|x \cap x'|}{|x \cup x'|} $`

### Hamming Distance (useful for strings!)

`$ d(x, x') = \frac{1}{n} \sum_{i=1}^n x_i \neq x'_i $`

</div>
</div>

<!--s-->

## Clustering | Properties

<div style="font-size: 0.75em;">

To determine the distance between two points (required for clustering), we need to define a distance metric. A good distance metric has the following properties:

1. Non-negativity: $d(x, y) \geq 0$
    - Otherwise, the distance between two points could be negative.
    - If Bob is 5 years old and Alice is 10 years old, the distance between them could be -5 years.

2. Identity: $d(x, y) = 0$ if and only if $x = y$
    - Otherwise, the distance between two points could be zero even if they are different.
    - If Bob is 5 years old and Alice is 5 years old, the distance between them should be zero.

3. Symmetry: $d(x, y) = d(y, x)$
    - Otherwise, the distance between two points could be different depending on the order.
    - If the distance between Bob and Alice is 5 years, the distance between Alice and Bob should also be 5 years.

4. Triangle inequality: $d(x, y) + d(y, z) \geq d(x, z)$
    - Otherwise, the distance between two points could be shorter than the sum of the distances between the points and a third point.
    - If the distance between Bob and Alice is 5 years and the distance between Alice and Charlie is 5 years, then the distance between Bob and Charlie should be at most 10 years.
</div>

<!--s-->

## Clustering Approaches

Okay, so we have a good understanding of distances. Now, let's talk about the different approaches to clustering data without labels using those distances. There are **many** clustering algorithms, but they can be broadly categorized into a few main approaches:


<img src="https://www.mdpi.com/sensors/sensors-23-06119/article_deploy/html/images/sensors-23-06119-g003.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Adnan, 2023</p>

<!--s-->

## Clustering Approaches

### Partitional Clustering
- Partitional clustering divides a dataset into $k$ clusters, where $k$ is a tunable hyperparameter.
- Example algorithm: K-Means

### Hierarchical Clustering
- Hierarchical clustering builds a hierarchy of clusters, which can be visualized as a dendrogram.
- Example algorithm: Agglomerative Clustering

### Density-Based Clustering
- Density-based clustering groups together points that are closely packed in the feature space.
- Example algorithm: DBSCAN

<!--s-->

<div class="header-slide">

# Partitional Clustering

</div>

<!--s-->

## Partitional Clustering | Introduction

Partitional clustering is the task of dividing a dataset into $k$ clusters, where $k$ is a hyperparameter. The goal is to minimize the intra-cluster distance and maximize the inter-cluster distance, subject to the constraint that each data point belongs to exactly one cluster.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vEsw12wO0KxvYne0m4Cr5w.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Yehoshua, 2023</p>

<!--s-->

## Partitional Clustering | K-Means

K-Means is a popular partitional clustering algorithm that partitions a dataset into $k$ clusters. The algorithm works as follows:

```text
1. Initialize $k$ cluster centroids randomly.
2. Assign each data point to the nearest cluster centroid.
3. Update the cluster centroids by taking the mean of the data points assigned to each cluster.
4. Repeat steps 2 and 3 until convergence.
    - Convergence occurs when the cluster centroids do not change significantly between iterations.
```


<img src="https://ben-tanen.com/assets/img/posts/kmeans-cluster.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Tanen, 2016</p>

<!--s-->

## Partitional Clustering | K-Means

K-Means is a simple and efficient algorithm, but it has some limitations:

- The number of clusters $k$ must be specified in advance.
- The algorithm is sensitive to the initial cluster centroids.
- The algorithm may converge to a local minimum.

There are variations of K-Means that help address limitations (e.g., K-Means++).

Specifying the number of clusters $k$ is a common challenge in clustering, but there are techniques to estimate an optimal $k$, such as the **elbow method** and the **silhouette score**. Neither of these techniques is perfect, but they can be informative under the right conditions.

<!--s-->

## Partitional Clustering | K-Means Elbow Method

The elbow method is a technique for estimating the number of clusters $k$ in a dataset. The idea is to plot the sum of squared distances between data points and their cluster centroids as a function of $k$, and look for an "elbow" in the plot.

```text
1. Run K-Means for different values of $k$.
2. For each value of $k$, calculate the sum of squared distances between data points and their cluster centroids.
3. Plot the sum of squared distances as a function of $k$.
4. Look for an "elbow" in the plot, where the rate of decrease in the sum of squared distances slows down.
5. The number of clusters $k$ at the elbow is a good estimate.
```

<img src="https://media.licdn.com/dms/image/D4D12AQF-yYtbzPvNFg/article-cover_image-shrink_600_2000/0/1682277078758?e=2147483647&v=beta&t=VhzheKDjy7bEcsYyrjql3NQAUcTaMBCTzhZWSVVSeNg" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Etemadi, 2020</p>

<!--s-->

## Partitional Clustering | K-Means Silhouette Score

The silhouette score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The optimal silhouette score is 1, and the worst score is -1. When using the silhouette score to estimate the number of clusters $k$, we look for the value of $k$ that maximizes the silhouette score.

$$ s = \frac{b - a}{\max(a, b)} $$

Where: 
- $a$ is the mean distance between a sample and all other points in the same cluster.
- $b$ is the mean distance between a sample and all other points in the next nearest cluster.

Pseudocode for estimating $k$ using the silhouette score:
```text
1. Run K-Means for different values of $k$.
2. For each value of $k$, calculate the silhouette score.
3. Plot the silhouette score as a function of $k$.
4. Look for the value of $k$ that maximizes the silhouette score.
```

<!--s-->

## Silhouette Score Pseudocode

```text
1. Determine the number of data points, n.
2. Initialize an array silhouette_values with zeros, length n.

3. For each data point i in X:
    a. Calculate a(i): 
       - Determine the cluster of point i, cluster_i.
       - Compute the average distance from point i to all other points in cluster_i.
    
    b. Calculate b(i):
       - For each cluster_k that is not cluster_i:
           - Compute the average distance from point i to all points in cluster_k.
           - Update b(i) to the minimum of its current value and this new average distance.
   
    c. Compute silhouette value for point i:
       - Use the formula: silhouette_values[i] = (b(i) - a(i)) / max(a(i), b(i))

4. Calculate the overall silhouette score:
   - Compute the mean of all entries in silhouette_values.
```

<!--s-->

<div class="header-slide">

# Hierarchical Clustering

</div>

<!--s-->

## Hierarchical Clustering | Introduction

Hierarchical clustering builds a hierarchy of clusters. The hierarchy can be visualized as a dendrogram, which shows the relationships between clusters at different levels of granularity.

### Agglomerative Clustering
- Start with each data point as a separate cluster, and merge clusters iteratively. AKA "bottom-up" clustering.

### Divisive Clustering
- Start with all data points in a single cluster, and split clusters iteratively. AKA "top-down" clustering.

<!--s-->

## Hierarchical Clustering | Agglomerative Clustering

Agglomerative clustering is a bottom-up approach to clustering that starts with each data point as a separate cluster, and merges clusters iteratively based on a linkage criterion. The algorithm works as follows:

```text
1. Start with each data point as a separate cluster.
2. Compute the distance between all pairs of clusters.
3. Merge the two closest clusters.
4. Repeat steps 2-3 until the desired number of clusters is reached.
```
<img src = "https://www.knime.com/sites/default/files/public/6-what-is-clustering.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hayasaka, 2021</p>

<!--s-->

## Hierarchical Clustering | Agglomerative Clustering Linkage

The distance between clusters can be computed using different linkage methods, such as:

- **Single linkage** (minimum distance between points in different clusters)
- **Complete linkage** (maximum distance between points in different clusters)
- **Average linkage** (average distance between points in different clusters)

The choice of linkage method can have a significant impact on the clustering result.

<img src = "https://www.knime.com/sites/default/files/public/6-what-is-clustering.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hayasaka, 2021</p>

<!--s-->

## Hierarchical Clustering | Agglomerative Clustering

Agglomerative clustering is a flexible and intuitive algorithm, but it has some limitations:

- The algorithm is sensitive to the choice of distance metric and linkage method.
- The algorithm has a time complexity of $O(n^3)$, which can be slow for large datasets.

<!--s-->

<div class="header-slide">

# Density-Based Clustering

</div>

<!--s-->

## Density-Based Clustering | Introduction

Density-based clustering groups together points that are closely packed in the feature space. The idea is to identify regions of high density and separate them from regions of low density.

One popular density-based clustering algorithm is DBSCAN.

<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*WunYbbKjzdXvw73a4Hd2ig.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Seo, 2023</p>

<!--s-->

## Density-Based Clustering | DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed in the feature space. The algorithm works as follows:

```text
1. For a data point, determine the neighborhood of points within a specified radius $\epsilon$.
2. If the neighborhood contains at least $m$ points, mark the data point as a core point.
3. Expand the cluster by adding all reachable points to the cluster.
4. Repeat steps 1-3 until all points have been assigned to a cluster.
```

DBSCAN has two hyperparameters: $\epsilon$ (the radius of the neighborhood) and $m$ (the minimum number of points required to form a cluster). The algorithm is robust to noise and can identify clusters of arbitrary shape.

<!--s-->

## Density-Based Clustering | DBSCAN

DBSCAN is a powerful algorithm for clustering data, but it has some limitations:

- Sensitive to the choice of hyperparameters $\epsilon$ and $m$.
- May struggle with clusters of varying densities.

<!--s-->

## Summary

- Unsupervised machine learning is about finding patterns in $X$, without being given $y$.
- Clustering is a fundamental technique in unsupervised machine learning, with a wide range of applications.
    - **Partitional clustering** divides a dataset into $k$ clusters, with K-Means being a popular algorithm.
    - **Hierarchical clustering** builds a hierarchy of clusters, with agglomerative clustering being a common approach.
    - **Density-based clustering** groups together points that are closely packed in the feature space, with DBSCAN being a popular algorithm.

<!--s-->

<div class="header-slide">

# Recommendation Systems

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with **recommendation** algorithms?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Intro Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

## Recommendation Systems

We will explore two common methods for building recommendation systems:

1. Content-Based Filtering
    - User Profile
    - Item Profile
    - Similarity Metrics
    - Evaluation
    - Pros and Cons
    
2. Collaborative Filtering
    - User-User
    - Item-Item
    - Filling in the Sparse Matrix
    - Pros and Cons

<!--s-->

## Recommendation Systems

Why build a recommendation system? 

- **Personalization**: Users are more likely to engage with a platform that provides personalized recommendations. Spotify absolutely excels at this.

- **Increased Engagement**: Users are more likely to spend more time on a platform that provides relevant recommendations. Instagram & TikTok are great examples of this.

- **Increased Revenue**: Users are more likely to purchase items that are recommended to them. According to a report by Salesforce, you can increase conversion rates for web products by 22.66% by using an intelligent recommendation system [[source]](https://brandcdn.exacttarget.com/sites/exacttarget/files/deliverables/etmc-predictiveintelligencebenchmarkreport.pdf).

<!--s-->

<div class="header-slide">

# Content-Based Filtering

</div>

<!--s-->

## Content-Based Filtering

Content-based filtering methods are based on a description of the **item** and a profile of the **user**’s preferences. A few quick definitions:

### User Matrix

- A set of data points that represents all user’s preferences. When a user interacts with an item, the user matrix is updated.

### Item Matrix

- A set of data points that represents every item’s content.
- Items can be represented as a set of features.
  - For books, these features could be genre, author, and publication date. More sophisticated models can use text analysis to extract features from the book's content.

### User-Item Similarity Matrix

- A set of data points that represents the similarity between user preferences and items.
- These similarities can be weighted by previous reviews or ratings from the user.

<!--s-->

## Content-Based Filtering | Basic Idea

The basic idea behind content-based filtering is to recommend items that are similar to those that a user liked in the past.

<img src="https://cdn.sanity.io/images/oaglaatp/production/a2fc251dcb1ad9ce9b8a82b182c6186d5caba036-1200x800.png?w=1200&h=800&auto=format" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Rosidi</p>

<!--s-->

## Content-Based Filtering | Intuitive Example & Question 

Let's say we have a user who has rated the following movies:

- The Shawshank Redemption: <span class="code-span">5</span>
- The Godfather: <span class="code-span">4</span>

And we want to know if they will enjoy:

- The Dark Knight: <span class="code-span">?</span>
- The Notebook: <span class="code-span">?</span>

We have the following information about the movies. They are encoded as an ordered, binary array of genres:

<span class="code-span">[ Drama, Crime, Thriller, Action, Romance]</span>

- The Shawshank Redemption: <span class="code-span">Drama, Crime, Thriller</span> | <span class="code-span"> [1, 1, 1, 0, 0]</span>
- The Godfather: <span class="code-span">Drama, Crime</span> | <span class="code-span"> [1, 1, 0, 0, 0]</span>
- The Dark Knight: <span class="code-span">Drama, Crime, Action</span> | <span class="code-span"> [1, 1, 0, 1, 0]</span>
- The Notebook: <span class="code-span">Drama, Romance</span> | <span class="code-span"> [1, 0, 0, 0, 1]</span>

Using content-based filtering, should we recommend (A) The Dark Knight or (B) The Notebook to the user?

<!--s-->

## Content-Based Filtering | Similarity Metrics

Commonly, similarity is determined simply as the dot product between two vectors:

$$ \textbf{Similarity} = A \cdot B $$

But exactly as represented by our KNN lecture, we can use different similarity metrics to calculate the similarity between two vectors.


<!--s-->

## Content-Based Filtering | Similarity / Distance Metrics

<div class = "col-wrapper">

<div class="c1" style = "width: 50%;">

### Euclidean Distance

`$ d(x, x') =\sqrt{\sum_{i=1}^n (x_i - x'_i)^2} $`

### Manhattan Distance

`$ d(x, x') = \sum_{i=1}^n |x_i - x'_i| $`

### Cosine Distance

`$ d(x, x') = 1 - \frac{x \cdot x'}{||x|| \cdot ||x'||} $`

</div>

<div class="c2" style = "width: 50%;">

### Jaccard Distance

`$ d(x, x') = 1 - \frac{|x \cap x'|}{|x \cup x'|} $`

### Hamming Distance

`$ d(x, x') = \frac{1}{n} \sum_{i=1}^n x_i \neq x'_i $`

</div>
</div>

<!--s-->

## Content-Based Filtering | Evaluation

Evaluation of content-based filtering depends on the application. Ideally, you would use a prospective study (A/B testing) to see if your recommendations are leading to increased user engagement!

<div class="col-centered">
<img src = "https://blog.christianposta.com/images/abtesting.png" style="border-radius:15px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Optimizely, 2024</p>
</div>

<!--s-->

## Content-Based Filtering | Pros and Cons

| Pros | Cons |
| --- | --- |
| Easy to implement. | Limited to the user’s past preferences. |
| No need for data on other users. | Limited to the item’s features. |
| Can recommend niche items. | Can overfit to the user’s preferences. |
| Can provide explanations for recommendations. | Cold-start problem. |

<!--s-->

## L06 | Q01

Which of the following is an advantage of content-based filtering?


<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. Easy to implement. <br>
&emsp;B. No need for data on other users. <br>
&emsp;C. Can recommend niche items. <br>
&emsp;D. All of the above.* <br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=L06 | Q01" width="100%" height="100%" style="border-radius: 10px;"></iframe>
</div>
</div>

<!--s-->

<div class="header-slide">

# Collaborative Filtering

</div>

<!--s-->

## Collaborative Filtering

### User-User Collaborative Filtering

Based on the idea that users who have "agreed in the past will agree in the future".

### Item-Item Collaborative Filtering

Based on the idea that item reviews are often grouped together.

<img src="https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-2.png" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NVIDIA, 2024</p>

<!--s-->

## Collaborative Filtering | User-Item Matrix

The User-Item Matrix is a matrix that represents the relationship between users and items.

You can find nearest neighbors from two different perspectives: user-user (rows) or item-item (cols).

<div class="col-centered">
<img src="https://storage.googleapis.com/slide_assets/user-item-matrix.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hashemi, 2020</p>
</div>

<!--s-->

## Collaborative Filtering | User-User

User-User Collaborative Filtering is based on the idea that users who have agreed in the past will agree in the future.

1. Create a User-Item Matrix that contains the user’s reviews of items.
2. Find users who are similar to the target user (User-User)
3. Recommend items that similar users have liked.

----------------

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<img src="https://storage.googleapis.com/slide_assets/user-item-matrix.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hashemi, 2020</p>

</div>
<div class="c2" style = "width: 50%">

<img src="https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-2.png" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NVIDIA, 2024</p>

</div>
</div>

<!--s-->

## Collaborative Filtering | Item-Item

Item-Item Collaborative Filtering is based on the idea that users who each liked an item will like similar items. It is **different** from content-based filtering because it does not have anything to do with the item characteristics.

1. Create a User-Item Matrix that represents the user’s reviews of items.
2. Find items that are often "grouped" together (Item-Item)
3. Recommend these similar items to the target user.

-------

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<img src="https://storage.googleapis.com/slide_assets/user-item-matrix.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hashemi, 2020</p>

</div>
<div class="c2" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:801/1*skK2fqWiBF7weHU8SjuCzw.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Tomar, 2017</p>

</div>
</div>

<!--s-->

## Collaborative Filtering
### Filling in the User-Item Matrix

The user-item matrix is often missing many values. This is because users typically only interact with a small subset of items. This presents a problem with finding nearest neighbors. 

So, how do we fill in the gaps?

<!--s-->

## Collaborative Filtering
### Filling in the User-Item Matrix with the Scale's Mean

A simple way to fill in the gaps in the user-item matrix is to use the scale's mean. Typically you will center the data on the mean of the scale, and fill in the gaps with zero.

In the example below, the scale is from 1-5. The mean of the scale is 3. So, we subtract 3 from every existing score and replace the rest with 0.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### Missing Values

|  | I.1 | I.2 | I.3 |
| --- | --- | --- | --- |
| U.1 | 5 | ... | 3 |
| U.2 | 0 | 4 | ... |
| U.3 | 3 | 2 | 0 |

</div>
<div class="c2" style = "width: 50%">

### Filled Values

|   | I.1 | I.2 | I.3 |
| --- | --- | --- | --- |
| U.1 | 2 | 0 | 0 |
| U.2 | -3 | 1 | 0 |
| U.3 | 0 | -1 | -3 |


</div>
</div>

<!--s-->

## Collaborative Filtering

### Filling in the User-Item Matrix with Matrix Factorization

Matrix Factorization is a technique to break down a matrix into the product of multiple matrices. It is used in collaborative filtering to estimate the missing values in the user-item matrix, and is often performed with alternating least squares (ALS).

$$ R \approx P \cdot Q^T $$

Where: 

- $R$ is the user-item matrix.
- $P$ is the user matrix.
- $Q$ is the item matrix.
- $K$ is the number of latent features.

<img src="https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/recommendation-system/img-6.png" height="30%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NVIDIA, 2024</p>

<!--s-->

## Collaborative Filtering

**Alternating Least Squares** is an optimization algorithm that is used to minimize the error between the predicted and actual ratings in the user-item matrix. There are many ways to solve for $P$ and $Q$! The example shown below is using the closed-form solution.

<div class = "col-wrapper">

<div class="c1" style = "width: 50%">

### Steps

1. Initialize the user matrix $P$ and the item matrix $Q$ randomly.

2. Fix $Q$ and solve for $P$.

3. Fix $P$ and solve for $Q$.

4. Repeat steps 2 and 3 until the error converges.

</div>

<div class="c2" style = "width: 50%">

### Example

$$ P = (Q^T \cdot Q + \lambda \cdot I)^{-1} \cdot Q^T \cdot R $$
$$ Q = (P^T \cdot P + \lambda \cdot I)^{-1} \cdot P^T \cdot R $$

Where:

- $R$ is the user-item matrix.
- $P$ is the user matrix.
- $Q$ is the item matrix.
- $\lambda$ is the regularization term.
- $I$ is the identity matrix.

<div style = "font-size: 0.6em;">

\**Note: You need to include the regularization term to prevent overfitting.*

</div>

<!--s-->

## Collaborative Filtering | Pros and Cons

| Pros | Cons |
| --- | --- |
| Can recommend items that the user has not seen before | Cold-start problem for both users and items |
| Can recommend items that are popular among similar users | Missing values in the user-item matrix |


<!--s-->

<div class="header-slide">

# Conclusion

</div>

<!--s-->

## Conclusion

In today’s lecture, we explored:

1. Content-Based Filtering
2. Collaborative Filtering

Recommendation systems are a powerful tool for personalizing user experiences and increasing user engagement.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
    <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with the **recommendation** algorithms?

  </div>
  </div>
  <div class="c2" style="width: 50%; height: 100%;">
  <iframe src="https://drc-cs-9a3f6.firebaseapp.com/?label=Exit Poll" width="100%" height="100%" style="border-radius: 10px"></iframe>
  </div>

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 1em; left: 0; width: 55%; position: absolute;">

  ## H4 | clustering.py
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