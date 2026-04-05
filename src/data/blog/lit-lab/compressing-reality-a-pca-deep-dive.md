---
author: Malaika Noor
pubDatetime: 2026-04-05T16:40:00.000Z
modDatetime: 2026-04-05T16:40:00.000Z
title: 'Compressing Reality: A PCA Deep Dive'
featured: true
draft: false
tags:
  - pca
  - principal component analysis
  - dimensionality reduction
  - math intuition
  - proof
  - pca implementation
ogImage: ''
description: A practical guide to Principal Component Analysis (PCA) covering the intuition, mathematical foundations, and a step-by-step implementation. Learn how PCA performs dimensionality reduction and why it is widely used in machine learning and data analysis.
hideEditPost: false
timezone: ''
---

**Principal Component Analysis** is a dimensionality reduction technique commonly used in machine learning, or in simple words, a method to represent inherently complex data in a simpler form.

#### The Problem

Assume you are training a model to detect fraud in credit card transactions, you are given some data which includes features like name, country, timestamp, amount, transaction type (online or physical), and the target which is whether a transaction is fraud or not. You start with EDA and realize that many features show a linear relationship with the target, e.g., 97% of fraudulent transactions are committed online and 99% of fraudulent transaction amounts were higher than a user's average transaction amount. Now you want your model to be very efficient, focusing on useful information and ignoring noise and redundancy. Practically, if a problem involves few features like less than 20, one may consider manually going through them and deciding which ones to keep, but in real world scenarios this is rarely the case. Data is usually messy with a lot of noise and redundancy and manual analysis is not practical.

![](https://miro.medium.com/v2/0*jEIuEzHO-kIop-6- "Direction of Variance - Courtesy: Medium")

This is where PCA comes in. We tell PCA how much information we want to keep and it converts our data to a new representation with smaller size. It works by capturing the directions of maximum variance in our data and ranking them in descending order. The data is then projected onto these new directions and we are left with a transformed dataset ready for modeling. Now let's look at how PCA actually achieves that.

---

#### The Requisites

Say our dataset is of shape $m \times n$, we represent it as matrix $X$, and there is another matrix $P$ which contains the principal components (the new directions) such that when multiplied to $X$, it transforms $X$ to $Y$, our desired form. More formally:

$$PX = Y$$

Matrices are linear transformations and from linear algebra we know that these transformations can change the basis of a space. Our data is nothing but a bunch of vectors in some space with some basis, and what we want is to convert this data to a reduced form with a new basis, specifically an orthonormal one where directions are perpendicular to each other and of unit length. A quick note on basis since it comes up here: a basis of a vector space is a set of vectors that span the space (any vector in the space can be written as a combination of them) and are linearly independent, so they form the minimum set of vectors needed to represent the space.

![](https://cdn-images-1.medium.com/v2/resize:fit:720/1*Q2hQIdy2G-lVSYGzy9W2lw.png "Change of Basis - Courtesy: Master the Math")

---

#### Variance and Covariance

Now to actually find this new basis, we need a way to represent variance in our data, especially when there is redundancy between features. This leads us to covariance, which captures how features vary together. We represent this using the covariance matrix, where each entry corresponds to the covariance between two features:

$$C_X = \frac{1}{m} XX^T$$

What we want is a new representation $Y$ such that its covariance matrix $C_Y$ has non-zero diagonal entries (variance of each component) and zero off-diagonal entries (no redundancy between components). In other words, we want $C_Y$ to be diagonal.

![](https://www.theanalysisfactor.com/wp-content/uploads/2011/09/covariance-matrix-1.png "Covariance matrix - Courtesy: The Analysis Factor")

---

#### The Proof

Starting from:

$$C_Y = \frac{1}{m} YY^T$$

Substituting $Y = PX$:

$$C_Y = \frac{1}{m}(PX)(PX)^T = \frac{1}{m}PXX^TP^T = PC_XP^T$$

Now comes the interesting part. $C_X$ is symmetric, and by the spectral decomposition theorem any symmetric matrix $A$ can be decomposed as:

$$A = EDE^T$$

where $E$ is orthogonal and contains the eigenvectors of $A$, and $D$ is diagonal and contains the corresponding eigenvalues. If we set $A = C_X$ and choose $P = E$, that is, let the rows of $P$ be the eigenvectors of $C_X$:

$$C_Y = E(EDE^T)E^T = (EE^T)D(EE^T) = D$$

since $EE^T = I$ by orthogonality. So $C_Y = D$, a diagonal matrix. We have proved that the covariance matrix of $Y$ captures the directions of maximum variance with zero redundancy, and the eigenvectors of $C_X$ are the principal components.

---

#### Implementation

Now as we have the intuition, we can move towards the implementation. We first center the data by subtracting the mean of each feature, because we want the spread of data measured relative to its own center, which in most cases is not already zero. Think of it this way: imagine you want to find the direction in which a crowd of people is most spread out in a field, but you measure everything relative to a random flag planted at $(0, 0)$ instead of the center of the crowd. You would get a distorted picture.

![](https://www.wangxinliu.com/images/machine_learning/original_data.png "Mean Centred Data - Courtesy: Wangxin's Blog")

After centering, we compute the covariance matrix $C_X$:

```plain
covariance = np.dot(X_centered.T, X_centered) / (self.n_samples_ - 1)
```

Then we compute the eigenvectors and eigenvalues of $C_X$ and sort them by eigenvalue in descending order, so the directions of highest variance come first:

```plain
eigvals, eigvecs = np.linalg.eigh(covariance)
sorting = np.argsort(eigvals)[::-1]
eigvals = eigvals[sorting]
eigvecs = eigvecs[:, sorting]
```

We then select the top $k$ components we want to retain:

```plain
components = eigvecs[:, :n_components].T
explained_variance = eigvals[:n_components]
total_var = eigvals.sum()
```

These components form the matrix $P$ which is multiplied to the original centered data to get the reduced representation $Y$.

![](/images/pca.png "Data projected on the PCA plane")

A complete implementation of PCA in python can be found [here](https://github.com/noor-malaika/litlab/blob/main/papers/pca/main.ipynb). 

Acknowledgement: [A Tutorial on Principal Component Analysis (Shlens 2014)](https://arxiv.org/pdf/1404.1100).
